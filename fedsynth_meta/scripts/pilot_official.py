#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FedSynthCT pilot using reviewed ModalityConverter mask semantics.

MONAI area-resizes the FOV mask as floating point. The official extension runs
an orthogonal slice only when the slice contains an exact value of 1 and sets
background only where the processed mask is exactly 0. Fractional edge values
remain part of the processed FOV. This module reproduces that behavior.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

import pilot_area as pa

MODEL_SHA256 = pa.MODEL_SHA256
ANALYSIS_VERSION = "FedSynthCT-Li-ONNX-reanalysis-v1.2.0-official-mask"
file_sha256 = pa.file_sha256
inspect_model = pa.inspect_model


def infer_planes_official(session, mri_cube: np.ndarray, mask_cube_float: np.ndarray):
    input_name = session.get_inputs()[0].name
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for axis, label in enumerate(("plane0", "plane1", "plane2")):
        volume = np.zeros(pa._pilot.CUBE, dtype=np.float32)
        started = time.perf_counter()
        for index in range(256):
            mri_slice = np.take(mri_cube, index, axis=axis)
            mask_slice = np.take(mask_cube_float, index, axis=axis)
            if not np.any(mask_slice == 1):
                continue
            model_input = mri_slice[None, None, :, :].astype(np.float32, copy=False)
            raw = session.run(None, {input_name: model_input})[0]
            predicted = np.asarray(raw).squeeze().astype(np.float32)
            if predicted.shape != (256, 256):
                raise RuntimeError(f"Unexpected ONNX output shape {predicted.shape}")
            if axis == 0:
                volume[index, :, :] = predicted
            elif axis == 1:
                volume[:, index, :] = predicted
            else:
                volume[:, :, index] = predicted
        volume[mask_cube_float == 0] = -1024.0
        outputs[label] = volume
        timings[label] = time.perf_counter() - started
    return outputs, timings


def process_case(case_dir: Path, session, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = case_dir.name
    center = case_id[2] if len(case_id) > 2 else "unknown"
    summary_dir = out / "summaries"
    figure_dir = out / "figures"
    volume_dir = out / "volumes" / case_id
    for directory in (summary_dir, figure_dir, volume_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mr_img = sitk.ReadImage(str(case_dir / "mr.nii.gz"))
    ct_img = sitk.ReadImage(str(case_dir / "ct.nii.gz"))
    mask_img = sitk.ReadImage(str(case_dir / "mask.nii.gz"))
    original_geometry = {
        "mr": pa._pilot.geometry(mr_img),
        "ct": pa._pilot.geometry(ct_img),
        "mask": pa._pilot.geometry(mask_img),
    }
    if not pa._pilot.same_geometry(ct_img, mr_img):
        ct_img = pa._pilot.resample_to_reference(ct_img, mr_img, False, -1024.0)
    if not pa._pilot.same_geometry(mask_img, mr_img):
        mask_img = pa._pilot.resample_to_reference(mask_img, mr_img, True, 0.0)

    corrected = pa._pilot.n4_correct(mr_img, mask_img)
    mr_native = sitk.GetArrayFromImage(corrected).astype(np.float32)
    ct_native = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask_native = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    tfm = pa._pilot.make_transform(tuple(int(x) for x in mr_native.shape))
    mr_cube = pa._pilot.normalize_mri(pa._pilot.to_cube(mr_native, tfm, 1, 0.0))
    ct_cube = pa._pilot.to_cube(ct_native, tfm, 1, -1024.0).astype(np.float32)
    mask_cube_float = pa._pilot.to_cube(mask_native, tfm, 0, 0.0).astype(np.float32)
    mask_cube = mask_cube_float != 0

    planes, timings = infer_planes_official(session, mr_cube, mask_cube_float)
    stack = np.stack([planes["plane0"], planes["plane1"], planes["plane2"]])
    fused = np.median(stack, axis=0).astype(np.float32)
    fused[mask_cube_float == 0] = -1024.0
    disagreement = np.ptp(stack, axis=0).astype(np.float32)
    mad = np.median(np.abs(stack - fused), axis=0).astype(np.float32)
    disagreement[mask_cube_float == 0] = 0.0
    mad[mask_cube_float == 0] = 0.0

    valid = mask_cube & np.isfinite(ct_cube)
    skull = valid & (ct_cube >= 200)
    regions = {
        "full_valid": valid,
        "soft_tissue": valid & (ct_cube >= -200) & (ct_cube <= 200),
        "skull_200": skull,
        "dense_bone_500": valid & (ct_cube >= 500),
        "bone_air_boundary": valid & (
            ndimage.binary_dilation(skull, iterations=2)
            ^ ndimage.binary_erosion(skull, iterations=2)
        ),
    }

    rows: list[dict[str, Any]] = []
    predictions = {**planes, "median_fusion": fused}
    for name, prediction in predictions.items():
        row: dict[str, Any] = {
            "case_id": case_id,
            "anonymous_center": center,
            "prediction": name,
        }
        for region_name, region in regions.items():
            row.update(pa._pilot.metric_block(ct_cube, prediction, region, region_name))
        row["bone_dice_200"] = pa._pilot.dice(ct_cube >= 200, prediction >= 200)
        row["bone_dice_500"] = pa._pilot.dice(ct_cube >= 500, prediction >= 500)
        row["slice_mean_ssim_clipped"] = pa._pilot.mean_slice_ssim(ct_cube, prediction, valid)
        row["psnr_clipped_valid"] = pa._pilot.psnr(ct_cube, prediction, valid)
        rows.append(row)

    absolute_error = np.abs(fused - ct_cube)
    rho_range, p_range, n_range = pa._pilot.qa_correlation(disagreement, absolute_error, valid)
    rho_mad, p_mad, n_mad = pa._pilot.qa_correlation(mad, absolute_error, valid)
    inside = fused[valid]
    percentiles = {str(q): float(np.percentile(inside, q)) for q in (0, 1, 50, 99, 100)}
    hu_gate = bool(inside.size and percentiles["1"] < -100 and percentiles["99"] > 200)

    for name, array in predictions.items():
        pa._pilot.save_volume(array, volume_dir / f"{case_id}_{name}_modelspace.nii.gz")
    pa._pilot.save_volume(disagreement, volume_dir / f"{case_id}_crossplane_range_modelspace.nii.gz")
    pa._pilot.save_volume(mad, volume_dir / f"{case_id}_crossplane_mad_modelspace.nii.gz")
    pa._pilot.make_panel(
        case_id,
        mr_cube,
        ct_cube,
        fused,
        absolute_error,
        disagreement,
        valid,
        figure_dir / f"{case_id}_pilot_panel.png",
    )

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": case_id,
        "anonymous_center": center,
        "claim_scope": "technical public-release model-space pilot; checkpoint-case overlap unknown",
        "mask_semantics": "MONAI area-resized float mask; exact-one slice activation; exact-zero background",
        "original_geometry": original_geometry,
        "analysis_transform": pa.asdict(tfm),
        "mask_fractional_voxels": int(np.sum((mask_cube_float > 0) & (mask_cube_float < 1))),
        "mask_nonzero_voxels": int(mask_cube.sum()),
        "timing_seconds_cpu": timings,
        "total_inference_seconds_cpu": float(sum(timings.values())),
        "fused_output_hu_percentiles": percentiles,
        "hu_plausibility_gate": hu_gate,
        "valid_voxels": int(valid.sum()),
        "qa": {
            "mean_range_hu": float(np.mean(disagreement[valid])),
            "p95_range_hu": float(np.percentile(disagreement[valid], 95)),
            "mean_mad_hu": float(np.mean(mad[valid])),
            "spearman_range_vs_abs_error": rho_range,
            "spearman_range_p": p_range,
            "spearman_range_n": n_range,
            "spearman_mad_vs_abs_error": rho_mad,
            "spearman_mad_p": p_mad,
            "spearman_mad_n": n_mad,
        },
    }
    (summary_dir / f"{case_id}_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    return summary, rows


def main() -> None:
    parser = pa._pilot.argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summaries").mkdir(exist_ok=True)

    actual_hash = file_sha256(args.model)
    if actual_hash != MODEL_SHA256:
        raise SystemExit(f"Model SHA-256 mismatch: {actual_hash}")
    model_info, session = inspect_model(args.model)
    model_info.update({
        "analysis_version": ANALYSIS_VERSION,
        "model_sha256": actual_hash,
        "python": pa._pilot.sys.version,
        "platform": pa._pilot.platform.platform(),
        "onnxruntime": pa._pilot.ort.__version__,
        "simpleitk": sitk.Version_VersionString(),
    })
    (args.out / "summaries" / "environment_and_model.json").write_text(
        json.dumps(model_info, indent=2), encoding="utf-8"
    )

    case_dirs = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / "mr.nii.gz").exists())
    if len(case_dirs) != 3:
        raise SystemExit(f"Expected three pilot cases, found {[p.name for p in case_dirs]}")

    summaries = []
    rows = []
    for case_dir in case_dirs:
        summary, case_rows = process_case(case_dir, session, args.out)
        summaries.append(summary)
        rows.extend(case_rows)

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summaries" / "pilot_metrics_long.csv", index=False)
    fused = table[table["prediction"] == "median_fusion"].copy()
    fused.to_csv(args.out / "summaries" / "pilot_metrics_median_fusion.csv", index=False)
    fused.select_dtypes(include=[np.number]).agg(["mean", "std", "min", "max"]).T.to_csv(
        args.out / "summaries" / "pilot_aggregate_median_fusion.csv"
    )
    overall = {
        "analysis_version": ANALYSIS_VERSION,
        "case_ids": [x["case_id"] for x in summaries],
        "anonymous_centers": [x["anonymous_center"] for x in summaries],
        "all_hu_plausibility_gates_passed": bool(all(x["hu_plausibility_gate"] for x in summaries)),
        "mean_full_valid_mae_hu": float(fused["full_valid_mae"].mean()),
        "mean_soft_tissue_mae_hu": float(fused["soft_tissue_mae"].mean()),
        "mean_skull_mae_hu": float(fused["skull_200_mae"].mean()),
        "mean_bone_dice_200": float(fused["bone_dice_200"].mean()),
        "mean_slice_ssim": float(fused["slice_mean_ssim_clipped"].mean()),
        "mean_psnr_db": float(fused["psnr_clipped_valid"].mean()),
        "mean_range_error_spearman": float(
            np.nanmean([x["qa"]["spearman_range_vs_abs_error"] for x in summaries])
        ),
        "limitations": [
            "Three-case technical pilot only.",
            "The public SynthRAD training release may overlap model training or tuning.",
            "The supplied mask is an MRI-derived evaluation/FOV mask, not a brain segmentation.",
            "Results are in model space; native-space parity remains unvalidated.",
            "Direct 3D Slicer parity remains a separate verification step.",
        ],
    }
    (args.out / "summaries" / "pilot_overall.json").write_text(
        json.dumps(overall, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
