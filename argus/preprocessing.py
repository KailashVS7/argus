from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tifffile


@dataclass
class ChannelStats:
    channel: str
    clip_low: float
    clip_high: float


@dataclass
class PreprocessMetadata:
    source: str
    input_mode: str
    channel_order: list[str]
    normalization: str
    stats: list[ChannelStats]


def _load_image_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        data = tifffile.imread(str(path))
    else:
        with Image.open(path) as img:
            if img.mode in {"L", "I", "F"}:
                data = np.asarray(img, dtype=np.float32)
            else:
                data = np.asarray(img.convert("RGB"), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _ensure_finite(name: str, array: np.ndarray) -> np.ndarray:
    if not np.isfinite(array).any():
        raise ValueError(f"{name} contains no finite values")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_channel(channel_name: str, channel: np.ndarray) -> tuple[np.ndarray, ChannelStats]:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        raise ValueError(f"{channel_name} contains no finite values")
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    if high <= low:
        high = low + 1e-6
    clipped = np.clip(channel, low, high)
    norm = (clipped - low) / (high - low)
    return norm.astype(np.float32), ChannelStats(channel_name, low, high)


def build_three_channel(
    *,
    vv: np.ndarray,
    vh: np.ndarray,
    source: str,
    input_mode: str,
) -> tuple[np.ndarray, PreprocessMetadata]:
    vv = _ensure_finite("vv", vv)
    vh = _ensure_finite("vh", vh)
    if vv.shape != vh.shape:
        raise ValueError(f"VV and VH shapes must match, got {vv.shape} vs {vh.shape}")

    derived = vv - vh
    vv_n, vv_stats = _normalize_channel("vv", vv)
    vh_n, vh_stats = _normalize_channel("vh", vh)
    d_n, d_stats = _normalize_channel("vv_minus_vh", derived)

    stacked = np.stack([vv_n, vh_n, d_n], axis=0)
    metadata = PreprocessMetadata(
        source=source,
        input_mode=input_mode,
        channel_order=["vv", "vh", "vv_minus_vh"],
        normalization="per_channel_percentile_clip_1_99_to_unit",
        stats=[vv_stats, vh_stats, d_stats],
    )
    return stacked, metadata


def load_vv_vh(
    *,
    vv_path: str | None,
    vh_path: str | None,
    input_path: str | None,
) -> tuple[np.ndarray, PreprocessMetadata]:
    if vv_path and vh_path:
        vv = _load_image_array(Path(vv_path))
        vh = _load_image_array(Path(vh_path))
        if vv.ndim == 3:
            vv = vv[..., 0]
        if vh.ndim == 3:
            vh = vh[..., 0]
        return build_three_channel(vv=vv, vh=vh, source=f"vv={vv_path},vh={vh_path}", input_mode="separate")

    if not input_path:
        raise ValueError("Provide --vv and --vh, or --input")

    arr = _load_image_array(Path(input_path))
    if arr.ndim == 2:
        vv = arr
        vh = arr
        mode = "single_band_demo_duplicate"
    elif arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        vv = arr[0, ...]
        vh = arr[1, ...]
        mode = "multi_band"
    elif arr.ndim == 3 and arr.shape[-1] >= 2:
        vv = arr[..., 0]
        vh = arr[..., 1]
        mode = "multi_band"
    else:
        raise ValueError(f"Unsupported input shape {arr.shape}")

    return build_three_channel(vv=vv, vh=vh, source=input_path, input_mode=mode)


def metadata_to_dict(metadata: PreprocessMetadata) -> dict[str, Any]:
    raw = asdict(metadata)
    raw["stats"] = [asdict(s) for s in metadata.stats]
    return raw


def save_preprocessed(output_npy: str, output_json: str, tensor: np.ndarray, metadata: PreprocessMetadata) -> None:
    np.save(output_npy, tensor)
    with Path(output_json).open("w", encoding="utf-8") as f:
        json.dump(metadata_to_dict(metadata), f, indent=2)
