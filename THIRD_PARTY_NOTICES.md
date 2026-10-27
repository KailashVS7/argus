# Third-Party Notices (Argus Proto 1)

Argus Proto 1 uses only free/open-source software components.

## Software

- **PyTorch** — BSD-3-Clause
  - https://github.com/pytorch/pytorch
- **TorchVision** — BSD-3-Clause
  - https://github.com/pytorch/vision
- **NumPy** — BSD-3-Clause
  - https://github.com/numpy/numpy
- **Pillow** — HPND (PIL Software License)
  - https://github.com/python-pillow/Pillow
- **tifffile** — BSD-3-Clause
  - https://github.com/cgohlke/tifffile

## Model Weights

- **VGG16 ImageNet weights** are provided by TorchVision model zoo when available locally or when explicitly downloaded by the user.
- Argus does **not** bundle model weights in this repository.
- Weight source metadata: `torchvision.models.VGG16_Weights.IMAGENET1K_V1`.

## Data

- This repository includes only locally generated synthetic demo fixtures.
- No external dataset artifacts are bundled.
- Users must verify and comply with dataset terms before running on real SAR scenes.
