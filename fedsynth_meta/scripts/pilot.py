#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned three-case technical pilot for FedSynthCT-Brain Li ONNX.

Research use only. This is a new ONNX reanalysis of public SynthRAD2023
training-release cases. It is not a reproduction of the July 2026 run and
makes no case-level independence claim. Metrics are computed in the documented
256 x 256 x 256 model space because native-space inverse parity is not yet
validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

MODEL_SHA256 = "52837d959b844bf83664d7e1e575d1a5f8cfbbec29bf1d52f3e59cc688d99fda"
ANALYSIS_VERSION = "FedSynthCT-Li-ONNX-reanalysis-v1.0.1"
CUBE = (256, 256, 256)
CROP_TARGET = (328, 256, 328)


@dataclass(frozen=True)
class Transform:
    original_shape: tuple[int, int, int]
    crop_start: tuple[int, int, int]
    crop_shape: tuple[int, int, int]
    resized_shape: tuple[int, int, int]
    pad_before: tuple[int, int, int]
    pad_after: tuple[int, int, int]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "pixel_type": image.GetPixelIDTypeAsString(),
    }


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )


def resample_to_reference(image: sitk.Image, reference: sitk.Image, nearest: bool, default: float) -> sitk.Image:
    filt = sitk.ResampleImageFilter()
    filt.SetReferenceImage(reference)
    filt.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
    filt.SetDefaultPixelValue(default)
    filt.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    return filt.Execute(image)


def n4_correct(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    source = sitk.Cast(image, sitk.sitkFloat32)
    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    small_image = sitk.Shrink(source, [2, 2, 2])
    small_mask = sitk.Shrink(binary, [2, 2, 2])
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 40, 30])
    corrector.SetConvergenceThreshold(1e-5)
    corrector.SetSplineOrder(3)
    corrector.Execute(small_image, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(source)
    corrected = source / sitk.Exp(log_bias)
    corrected.CopyInformation(image)
    return corrected


def make_transform(shape: tuple[int, int, int]) -> Transform:
    starts = []
    crop_shape = []
    for dim, wanted in zip(shape, CROP_TARGET):
        size = min(dim, wanted)
        starts.append(max((dim - size) // 2, 0))
        crop_shape.append(size)
    resized = tuple(min(dim, 256) for dim in crop_shape)
    before = tuple((256 - dim) // 2 for dim in resized)
    after = tuple(256 - dim - b for dim, b in zip(resized, before))
    return Transform(
        tuple(int(x) for x in shape),
        tuple(int(x) for x in starts),
        tuple(int(x) for x in crop_shape),
        tuple(int(x) for x in resized),
        tuple(int(x) for x in before),
        tuple(int(x) for x in after),
    )


def resize_array(array: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    if tuple(array.shape) == tuple(shape):
        return np.asarray(array).copy()
    zoom = tuple(new / old for new, old in zip(shape, array.shape))
    return ndimage.zoom(array, zoom=zoom, order=order, mode="nearest", prefilter=(order > 1))


def to_cube(array: np.ndarray, tfm: Transform, order: int, background: float) -> np.ndarray:
    slices = tuple(slice(start, start + size) for start, size in zip(tfm.crop_start, tfm.crop_shape))
    cropped = array[slices]
    resized = resize_array(cropped, tfm.resized_shape, order)
    pads = tuple(zip(tfm.pad_before, tfm.pad_after))
    cube = np.pad(resized, pads, mode="constant", constant_values=background)
    if cube.shape != CUBE:
        raise RuntimeError(f"Unexpected cube shape {cube.shape}")
    return cube


def normalize_mri(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid MRI range {lo}, {hi}")
    return np.nan_to_num((arr - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def inspect_model(path: Path) -> tuple[dict[str, Any], ort.InferenceSession]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    metadata = {
        "ir_version": int(model.ir_version),
        "opsets": [{"domain": x.domain, "version": int(x.version)} for x in model.opset_import],
        "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_inputs()],
        "outputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_outputs()],
        "providers": session.get_providers(),
    }
    return metadata, session


def infer_planes(session: ort.InferenceSession, mri: np.ndarray, mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    input_name = session.get_inputs()[0].name
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for axis, label in enumerate(("plane0", "plane1", "plane2")):
        volume = np.zeros(CUBE, dtype=np.float32)
        started = time.perf_counter()
        for idx in range(256):
            mri_slice = np.take(mri, idx, axis=axis)
            mask_slice = np.take(mask, idx, axis=axis)
            if not np.any(mask_slice):
                continue
            model_input = mri_slice[None, None, :, :].astype(np.float32, copy=False)
            raw = session.run(None, {input_name: model_input})[0]
            pred = np.asarray(raw).squeeze().astype(np.float32)
            if pred.shape != (256, 256):
                raise RuntimeError(f"Unexpected ONNX output shape {pred.shape}")
            if axis == 0:
                volume[idx, :, :] = pred
            elif axis == 1:
                volume[:, idx, :] = pred
            else:
                volume[:, :, idx] = pred
        volume[~mask] = -1024.0
        outputs[label] = volume
        timings[label] = time.perf_counter() - started
    return outputs, timings


def metric_block(reference: np.ndarray, prediction: np.ndarray, region: np.ndarray, prefix: str) -> dict[str, float]:
    valid = region & np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(valid):
        return {f"{prefix}_n": 0, f"{prefix}_mae": float("nan"), f"{prefix}_rmse": float("nan"), f"{prefix}_bias": float("nan")}
    diff = prediction[valid] - reference[valid]
    return {
        f"{prefix}_n": int(valid.sum()),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}_bias": float(np.mean(diff)),
    }


def dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, bool)
    bb = np.asarray(b, bool)
    denom = int(aa.sum() + bb.sum())
    return float(2 * np.logical_and(aa, bb).sum() / denom) if denom else float("nan")


def mean_slice_ssim(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    a = np.clip(reference, -1000.0, 2000.0)
    b = np.clip(prediction, -1000.0, 2000.0)
    vals = []
    for z in range(a.shape[0]):
        if int(mask[z].sum()) < 100:
            continue
        vals.append(float(structural_similarity(a[z], b[z], data_range=3000.0)))
    return float(np.mean(vals)) if vals else float("nan")


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    diff = np.clip(prediction[valid], -1000, 2000) - np.clip(reference[valid], -1000, 2000)
    mse = float(np.mean(diff ** 2))
    return float("inf") if mse == 0 else float(20 * np.log10(3000.0 / np.sqrt(mse)))


def qa_correlation(score: np.ndarray, error: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    valid = mask & np.isfinite(score) & np.isfinite(error)
    x = score[valid].ravel()
    y = error[valid].ravel()
    if x.size > 500000:
        rng = np.random.default_rng(20261124)
        ids = rng.choice(x.size, 500000, replace=False)
        x = x[ids]
        y = y[ids]
    if x.size < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), int(x.size)
    rho, p = spearmanr(x, y)
    return float(rho), float(p), int(x.size)


def save_volume(array: np.ndarray, path: Path) -> None:
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path), True)


def make_panel(case_id: str, mri: np.ndarray, ct: np.ndarray, sct: np.ndarray, error: np.ndarray, disagreement: np.ndarray, mask: np.ndarray, path: Path) -> None:
    valid_slices = np.where(mask.any(axis=(1, 2)))[0]
    z = int(np.median(valid_slices)) if valid_slices.size else 128
    data = [mri[z], ct[z], sct[z], error[z], disagreement[z]]
    titles = ["T1 MRI", "Reference CT", "Median sCT", "Absolute error", "Cross-plane range"]
    cmaps = ["gray", "gray", "gray", "magma", "viridis"]
    ranges = [(0, 1), (-100, 200), (-100, 200), (0, 300), (0, 300)]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    for ax, arr, title, cmap, limits in zip(axes, data, titles, cmaps, ranges):
        im = ax.imshow(arr, cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(title)
        ax.axis("off")
        if title in ("Absolute error", "Cross-plane range"):
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f"{case_id}: technical model-space pilot; no independence claim")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def process_case(case_dir: Path, session: ort.InferenceSession, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    original_geometry = {"mr": geometry(mr_img), "ct": geometry(ct_img), "mask": geometry(mask_img)}
    if not same_geometry(ct_img, mr_img):
        ct_img = resample_to_reference(ct_img, mr_img, False, -1024.0)
    if not same_geometry(mask_img, mr_img):
        mask_img = resample_to_reference(mask_img, mr_img, True, 0.0)

    corrected = n4_correct(mr_img, mask_img)
    mr_native = sitk.GetArrayFromImage(corrected).astype(np.float32)
    ct_native = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask_native = sitk.GetArrayFromImage(mask_img) > 0
    tfm = make_transform(tuple(int(x) for x in mr_native.shape))
    mr_cube = normalize_mri(to_cube(mr_native, tfm, 1, 0.0))
    ct_cube = to_cube(ct_native, tfm, 1, -1024.0).astype(np.float32)
    mask_cube = to_cube(mask_native.astype(np.float32), tfm, 0, 0.0) > 0.5

    planes, timings = infer_planes(session, mr_cube, mask_cube)
    stack = np.stack([planes["plane0"], planes["plane1"], planes["plane2"]])
    fused = np.median(stack, axis=0).astype(np.float32)
    fused[~mask_cube] = -1024.0
    disagreement = np.ptp(stack, axis=0).astype(np.float32)
    mad = np.median(np.abs(stack - fused), axis=0).astype(np.float32)
    disagreement[~mask_cube] = 0.0
    mad[~mask_cube] = 0.0

    valid = mask_cube & np.isfinite(ct_cube)
    skull = valid & (ct_cube >= 200)
    regions = {
        "full_valid": valid,
        "soft_tissue": valid & (ct_cube >= -200) & (ct_cube <= 200),
        "skull_200": skull,
        "dense_bone_500": valid & (ct_cube >= 500),
        "bone_air_boundary": valid & (ndimage.binary_dilation(skull, iterations=2) ^ ndimage.binary_erosion(skull, iterations=2)),
    }

    rows = []
    predictions = {**planes, "median_fusion": fused}
    for name, pred in predictions.items():
        row: dict[str, Any] = {"case_id": case_id, "anonymous_center": center, "prediction": name}
        for region_name, region in regions.items():
            row.update(metric_block(ct_cube, pred, region, region_name))
        row["bone_dice_200"] = dice(ct_cube >= 200, pred >= 200)
        row["bone_dice_500"] = dice(ct_cube >= 500, pred >= 500)
        row["slice_mean_ssim_clipped"] = mean_slice_ssim(ct_cube, pred, valid)
        row["psnr_clipped_valid"] = psnr(ct_cube, pred, valid)
        rows.append(row)

    absolute_error = np.abs(fused - ct_cube)
    rho_range, p_range, n_range = qa_correlation(disagreement, absolute_error, valid)
    rho_mad, p_mad, n_mad = qa_correlation(mad, absolute_error, valid)
    inside = fused[valid]
    percentiles = {str(q): float(np.percentile(inside, q)) for q in (0, 1, 50, 99, 100)}
    hu_gate = bool(inside.size and percentiles["1"] < -100 and percentiles["99"] > 200)

    for name, arr in predictions.items():
        save_volume(arr, volume_dir / f"{case_id}_{name}_modelspace.nii.gz")
    save_volume(disagreement, volume_dir / f"{case_id}_crossplane_range_modelspace.nii.gz")
    save_volume(mad, volume_dir / f"{case_id}_crossplane_mad_modelspace.nii.gz")
    make_panel(case_id, mr_cube, ct_cube, fused, absolute_error, disagreement, valid, figure_dir / f"{case_id}_pilot_panel.png")

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": case_id,
        "anonymous_center": center,
        "claim_scope": "technical public-release model-space pilot; checkpoint-case overlap unknown",
        "original_geometry": original_geometry,
        "analysis_transform": asdict(tfm),
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
    (summary_dir / f"{case_id}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "python": sys.version,
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "simpleitk": sitk.Version_VersionString(),
    })
    (args.out / "summaries" / "environment_and_model.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

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
        "mean_range_error_spearman": float(np.nanmean([x["qa"]["spearman_range_vs_abs_error"] for x in summaries])),
        "limitations": [
            "Three-case technical pilot only.",
            "The public SynthRAD training release may overlap with model training or tuning.",
            "The supplied mask is an evaluation/FOV mask, not a brain segmentation.",
            "Results are in model space; native-space parity remains unvalidated.",
            "The headless implementation approximates the reviewed Slicer preprocessing and requires direct parity testing.",
        ],
    }
    (args.out / "summaries" / "pilot_overall.json").write_text(json.dumps(overall, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(overall, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
