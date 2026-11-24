#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the FedSynthCT pilot with MONAI-compatible adaptive-area resizing.

The official SlicerModalityConverter CustomResize delegates to MONAI Resize
with its default area mode. PyTorch implements this as adaptive average
pooling. This module applies the same discrete start/end binning without
requiring a large Torch installation, then runs the versioned pilot.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pilot as _pilot


def _adaptive_mean_axis(array: np.ndarray, output_size: int, axis: int) -> np.ndarray:
    input_size = array.shape[axis]
    if input_size == output_size:
        return np.asarray(array).copy()
    moved = np.moveaxis(np.asarray(array, dtype=np.float32), axis, 0)
    pooled = []
    for index in range(output_size):
        start = (index * input_size) // output_size
        end = ((index + 1) * input_size + output_size - 1) // output_size
        pooled.append(moved[start:end].mean(axis=0, dtype=np.float32))
    return np.moveaxis(np.stack(pooled, axis=0), 0, axis)


def adaptive_area_resize(array: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    del order  # official MONAI Resize default is area for both MRI and mask
    result = np.asarray(array, dtype=np.float32)
    for axis, output_size in enumerate(shape):
        result = _adaptive_mean_axis(result, int(output_size), axis)
    return result.astype(np.float32, copy=False)


_pilot.resize_array = adaptive_area_resize
_pilot.ANALYSIS_VERSION = "FedSynthCT-Li-ONNX-reanalysis-v1.1.1-area-hotfix"

MODEL_SHA256 = _pilot.MODEL_SHA256
ANALYSIS_VERSION = _pilot.ANALYSIS_VERSION
file_sha256 = _pilot.file_sha256
inspect_model = _pilot.inspect_model
process_case = _pilot.process_case
asdict = _pilot.asdict

if __name__ == "__main__":
    _pilot.main()
