# FedSynthCT-Brain: versioned Li-ONNX technical reanalysis

This branch implements a **new versioned technical reanalysis** of the officially released FedSynthCT-Brain Li ONNX model. It does not reproduce the historical July 2026 run and does not establish case-level independence from model training or tuning.

The pilot:

1. range-extracts one paired Task1 brain MRI/CT case from each anonymous SynthRAD center A/B/C without downloading the complete 14.5-GB archive;
2. verifies the official Li-model SHA-256;
3. applies the reviewed N4, center-crop, resize, padding, min-max normalization, and three-plane inference workflow;
4. exports all three pre-fusion sCT predictions, the voxel-wise median, and cross-plane disagreement maps;
5. evaluates full-mask, soft-tissue, skull, dense-bone, and bone-air-interface errors in the documented 256³ model space;
6. reports fusion ablation and reference-free cross-plane disagreement versus paired-CT error.

## Scientific scope

The public Zenodo release contains SynthRAD training data. The relation between individual cases and the later ONNX checkpoint is not documed. Therefore, the pilot is a technical and quantitative validation of the released workflow, **not an independent external-validation claim**. The supplied SynthRAD mask is treated as an evaluation/FOV mask, not as a brain-parenchyma segmentation.

## Licenses

- SynthRAD2023 image data: CC BY-NC 4.0.
- SlicerModalityConverter code: Apache-2.0.
- Released ONNX models: AI Pubs research-use RAIL-M v0.1.

Research use only. No clinical, diagnostic, commercial, or treatment-planning use is asserted.
