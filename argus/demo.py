from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def generate_synthetic_demo(output_dir: str, size: int = 128) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    vv = (rng.normal(0.3, 0.08, (size, size))).astype(np.float32)
    vh = (rng.normal(0.25, 0.08, (size, size))).astype(np.float32)

    vv[30:38, 20:50] += 0.5
    vh[30:38, 20:50] += 0.2
    vv[75:84, 70:110] += 0.45
    vh[75:84, 70:110] += 0.15

    vv_path = out / "synthetic_vv.tif"
    vh_path = out / "synthetic_vh.tif"
    Image.fromarray((np.clip(vv, 0, 1) * 255).astype(np.uint8)).save(vv_path)
    Image.fromarray((np.clip(vh, 0, 1) * 255).astype(np.uint8)).save(vh_path)

    rgb = np.stack([
        np.clip(vv, 0, 1),
        np.clip(vh, 0, 1),
        np.clip(vv - vh + 0.5, 0, 1),
    ], axis=-1)
    rgb_path = out / "synthetic_demo_rgb.png"
    Image.fromarray((rgb * 255).astype(np.uint8)).save(rgb_path)

    return {
        "vv": str(vv_path),
        "vh": str(vh_path),
        "rgb_demo": str(rgb_path),
    }
