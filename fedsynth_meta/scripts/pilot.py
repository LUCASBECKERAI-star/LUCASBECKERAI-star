#!/usr/bin/env python3
"""Three-case technical pilot for the released FedSynthCT-Brain Li ONNX model.

This is a new, versioned ONNX reanalysis. It is not a reproduction of the
historical July 2026 analysis and does not establish case-level independence
from model training or tuning. All quantitative comparisons are performed in
the documented 256^3 model space, avoiding an unvalidated inverse transform to
native geometry.
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
ANALYSIS_VERSION = "FedSynthCT-Li-ONNX-reanalysis-v1.0.0"
MODEL_CUBE_SHAPE = (256, 256, 256)  # array order z, y, x
CROP_TARGET = (328, 256, 328)       # follows reviewed Slicer/MONAI implementation


@dataclass(frozen=True)
class SpatialTransform:
    original_shape: tuple[int, int, int]
    crop_start: tuple[int, int, int]
    crop_shape: tuple[int, int, int]
    resized_shape: tuple[int, int, int]
    pad_before: tuple[int, int, int]
    pad_after: tuple[int, int, int]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "pixel_type": image.GetPixelIDTypeAsString(),
    }


def same_geometry(a: sitk.Image, b: sitk.Image, tolerance: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tolerance)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tolerance)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tolerance)
    )


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    nearest: bool,
    default_value: float,
) -> sitk.Image:
    filt = sitk.ResampleImageFilter()
    filt.SetReferenceImage(reference)
    filt.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
    filt.SetDefaultPixelValue(default_value)
    filt.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    return filt.Execute(image)


def n4_bias_correct(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    source = sitk.Cast(image, sitk.sitkFloat32)
    binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    shrink = [2, 2, 2]
    small_image = sitk.Shrink(source, shrink)
    small_mask = sitk.Shrink(binary_mask, shrink)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 40, 30])
    corrector.SetConvergenceThreshold(1e-5)
    corrector.SetSplineOrder(3)
    _ = corrector.Execute(small_image, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(source)
    corrected = source / sitk.Exp(log_bias)
    corrected.CopyInformation(image)
    return corrected


def define_spatial_transform(shape: tuple[int, int, int]) -> SpatialTransform:
    starts: list[int] = []
    crop_shape: list[int] = []
    for dim, wanted in zip(shape, CROP_TARGET):
        size = min(dim, wanted)
        starts.append(max((dim - size) // 2, 0))
        crop_shape.append(size)
    resized_shape = tuple(min(dim, 256) for dim in crop_shape)
    pad_before = tuple((256 - dim) // 2 for dim in resized_shape)
    pad_after = tuple(256 - dim - before for dim, before in zip(resized_shape, pad_before))
    return SpatialTransform(
        original_shape=tuple(int(x) for x in shape),
        crop_start=tuple(int(x) for x in starts),
        crop_shape=tuple(int(x) for x in crop_shape),
        resized_shape=tuple(int(x) for x in resized_shape),
        pad_before=tuple(int(x) for x in pad_before),
        pad_after=tuple(int(x) for x in pad_after),
    )


def resize_array(array: np.ndarray, new_shape: tuple[int, int, int], order: int) -> np.ndarray:
    if tuple(array.shape) == tuple(new_shape):
        return np.array(array, copy=True)
    zoom = tuple(new / old for new, old in zip(new_shape, array.shape))
    return ndimage.zoom(array, zoom=zoom, order=order, mode="nearest", prefilter=order > 1)


def transform_to_cube(
    array: np.ndarray,
    transform: SpatialTransform,
    *,
    interpolation_order: int,
    background_value: float,
) -> np.ndarray:
    crop_slices = tuple(
        slice(start, start + size)
        for start, size in zip(transform.crop_start, transform.crop_shape)
    )
    cropped = array[crop_slices]
    resized = resize_array(cropped, transform.resized_shape, interpolation_order)
    pad_width = tuple(zip(transform.pad_before, transform.pad_after))
    cube = np.pad(resized, pad_width, mode="constant", constant_values=background_value)
    if cube.shape != MODEL_CUBE_SHAPE:
        raise RuntimeError(f"Unexpected model cube shape: {cube.shape}")
    return cube


def normalize_mri(cube: np.ndarray) -> np.ndarray:
    cube = cube.astype(np.float32, copy=False)
    minimum = float(np.nanmin(cube))
    maximum = float(np.nanmax(cube))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"Invalid MRI intensity range: {minimum}, {maximum}")
    normalized = (cube - minimum) / (maximum - minimum)
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def inspect_model(model_path: Path) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return {
        "ir_version": int(model.ir_version),
        "opsets": [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import],
        "inputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_inputs()],
        "outputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_outputs()],
        "providers": session.get_providers(),
    }


def infer_three_planes(
    session: ort.InferenceSession,
    mri_cube: np.ndarray,
    mask_cube: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    input_name = session.get_inputs()[0].name
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for axis, name in enumerate(("plane0", "plane1", "plane2")):
        prediction = np.zeros(MODEL_CUBE_SHAPE, dtype=np.float32)
        started = time.perf_counter()
        for index in range(256):
            mri_slice = np.take(mri_cube, index, axis=axis)
            mask_slice = np.take(mask_cube, index, axis=axis)
            if not np.any(mask_slice):
                continue
            model_input = mri_slice[None, None, :, :].astype(np.float32, copy=False)
            raw = session.run(None, {input_name: model_input})[0]
            predicted_slice = np.asarray(raw).squeeze().astype(np.float32)
            if predicted_slice.shape != (256, 256):
                raise RuntimeError(f"Unexpected ONNX output shape {predicted_slice.shape}")
            if axis == 0:
                prediction[index, :, :] = predicted_slice
            elif axis == 1:
                prediction[:, index, :] = predicted_slice
            else:
                prediction[:, :, index] = predicted_slice
        prediction[~mask_cube] = -1024.0
        timings[name] = time.perf_counter() - started
        outputs[name] = prediction
    return outputs, timings


def region_metrics(reference: np.ndarray, prediction: np.ndarray, region: np.ndarray, prefix: str) -> dict[str, float]:
    valid = region & np.isfinite(reference) & np.isfinite(prediction)
    count = int(valid.sum())
    if count == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_bias": float("nan"),
        }
    difference = prediction[valid] - reference[valid]
    return {
        f"{prefix}_n": count,
        f"{prefix}_mae": float(np.mean(np.abs(difference))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(difference**2))),
        f"{prefix}_bias": float(np.mean(difference)),
    }


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    denominator = int(aa.sum() + bb.sum())
    return float(2 * np.logical_and(aa, bb).sum() / denominator) if denominator else float("nan")


def slice_mean_ssim(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    values: list[float] = []
    clipped_reference = np.clip(reference, -1000.0, 2000.0)
    clipped_prediction = np.clip(prediction, -1000.0, 2000.0)
    for index in range(reference.shape[0]):
        if int(mask[index].sum()) < 100:
            continue
        values.append(
            float(
                structural_similarity(
                    clipped_reference[index],
                    clipped_prediction[index],
                    data_range=3000.0,
                )
            )
        )
    return float(np.mean(values)) if values else float("nan")


def psnr_in_mask(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(reference) & np.isfinite(prediction)
    if not valid.any():
        return float("nan")
    difference = np.clip(prediction[valid], -1000, 2000) - np.clip(reference[valid], -1000, 2000)
    mse = float(np.mean(difference**2))
    return float("inf") if mse == 0 else float(20 * np.log10(3000.0 / np.sqrt(mse)))


def sampled_spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray, maximum: int = 500_000) -> tuple[float, float, int]:
    valid = mask & np.isfinite(x) & np.isfinite(y)
    xx = x[valid].ravel()
    yy = y[valid].ravel()
    if xx.size > maximum:
        rng = np.random.default_rng(20261124)
        indices = rng.choice(xx.size, maximum, replace=False)
        xx = xx[indices]
        yy = yy[indices]
    if xx.size < 10 or np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return float("nan"), float("nan"), int(xx.size)
    coefficient, p_value = spearmanr(xx, yy)
    return float(coefficient), float(p_value), int(xx.size)


def save_model_space_nifti(array: np.ndarray, path: Path) -> None:
    image = sitk.Ge}õÛ]œ…ªì¶»§q«^