# argus

Argus Proto 1 is an **offline, CPU-compatible baseline** for vessel detection from Sentinel-1 style SAR inputs around Palk Bay.

## Scope of Proto 1

- ✅ VV/VH/derived 3-channel preprocessing (`VV`, `VH`, `VV - VH`)
- ✅ VGG16 backbone + Faster R-CNN baseline
- ✅ Horizontal bounding box outputs only
- ✅ CLI-first workflow; optional local lightweight GUI via tkinter
- ✅ Output artifacts: annotated image + JSON + CSV
- ❌ No AIS matching
- ❌ No ghost-vessel claim logic
- ❌ No rotated bounding boxes yet (planned Stage 2)

> **Safety note:** detections from this prototype are **not proof of ghost vessels** and should not be used for enforcement decisions.

## Hardware/OS target

Designed for Linux CPU inference (including Intel i7 class systems without CUDA).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

## Offline and weights behavior

Argus supports explicit weight behavior:

- `--weights none`: never tries to load ImageNet weights.
- `--weights imagenet`: request VGG16 ImageNet initialization.
- `--no-download`: if ImageNet weights are not already cached locally, Argus falls back to random-init backbone and reports that behavior.

This avoids forced network access during tests/offline runs.

## Quick deterministic demo (offline)

```bash
argus make-demo --output-dir ./demo
argus infer \
  --vv ./demo/synthetic_vv.tif \
  --vh ./demo/synthetic_vh.tif \
  --weights none \
  --no-download \
  --output-dir ./outputs \
  --score-threshold 0.99
```

Outputs in `./outputs`:

- `annotated.png`
- `detections.json`
- `detections.csv`

The synthetic demo is a smoke test only and does **not** represent real SAR accuracy.

## CLI commands

```bash
argus make-demo --output-dir <dir>
argus preprocess --vv <vv.tif> --vh <vh.tif> --output-npy <tensor.npy> --output-json <meta.json>
argus preprocess --input <2-band.tif|image.png> --output-npy <tensor.npy> --output-json <meta.json>
argus init-model --weights {none,imagenet} [--no-download] --checkpoint <ckpt.pt>
argus infer [--vv <vv.tif> --vh <vh.tif> | --input <image>] \
  --weights {none,imagenet} [--no-download] [--checkpoint <ckpt.pt>] \
  --output-dir <dir> [--score-threshold 0.5]
argus gui
```

## Input behavior

- Preferred: separate `VV` and `VH` files.
- Supported: 2-band TIFF/GeoTIFF style raster.
- Demo-only fallback: single-band or RGB image is mapped to pseudo VV/VH (documented in metadata); this is not true SAR.

## Performance expectations

Proto 1 targets correctness and reproducibility first. CPU inference on integrated-graphics laptops may be slow for large rasters; use smaller tiles/chips for local demos.

## Data and licensing guidance

- Use only openly licensed/permitted data.
- Do not use paid, credential-gated, or non-free data/services for this prototype.
- Verify terms for any third-party dataset before use.

## Limitations

- Generic ImageNet-pretrained VGG16 is domain-mismatched for SAR vessel detection.
- Meaningful SAR inference requires a compatible SAR-trained checkpoint.
- No geospatial post-processing, AIS matching, or surveillance claims are included.

## Stage 2 roadmap

- Rotated bounding-box detection for elongated vessels.
- Optional SAR-specific backbone/checkpoints.
- Structured geospatial export improvements.
