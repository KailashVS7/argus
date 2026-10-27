from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .preprocessing import load_vv_vh, metadata_to_dict
from .serialization import save_predictions_csv, save_predictions_json


CLASS_MAP = {1: "vessel"}


def run_inference(
    *,
    model: torch.nn.Module,
    vv_path: str | None,
    vh_path: str | None,
    input_path: str | None,
    output_dir: str,
    score_threshold: float = 0.5,
) -> dict:
    tensor_np, prep_meta = load_vv_vh(vv_path=vv_path, vh_path=vh_path, input_path=input_path)
    image_tensor = torch.from_numpy(tensor_np)

    model.eval()
    with torch.no_grad():
        output = model([image_tensor])[0]

    boxes = output.get("boxes", torch.empty((0, 4))).cpu().numpy()
    scores = output.get("scores", torch.empty((0,))).cpu().numpy()
    labels = output.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy()

    src = input_path or f"vv={vv_path},vh={vh_path}"
    rows: list[dict] = []
    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_threshold:
            continue
        rows.append(
            {
                "source_image": src,
                "class": CLASS_MAP.get(int(label), f"class_{int(label)}"),
                "confidence": float(score),
                "xmin": float(box[0]),
                "ymin": float(box[1]),
                "xmax": float(box[2]),
                "ymax": float(box[3]),
            }
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = np.transpose(tensor_np, (1, 2, 0))
    preview = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    annotated = Image.fromarray(preview)
    draw = ImageDraw.Draw(annotated)
    for row in rows:
        draw.rectangle([row["xmin"], row["ymin"], row["xmax"], row["ymax"]], outline="red", width=2)
        draw.text((row["xmin"], max(0, row["ymin"] - 10)), f"{row['class']}:{row['confidence']:.2f}", fill="red")
    annotated_path = out / "annotated.png"
    annotated.save(annotated_path)

    payload = {
        "source_image": src,
        "detections": rows,
        "preprocessing": metadata_to_dict(prep_meta),
        "notes": [
            "Proto 1 baseline with horizontal boxes only.",
            "Detections are not proof of ghost vessels and are not AIS-matched.",
            "Meaningful SAR performance requires a compatible SAR-trained checkpoint.",
        ],
    }

    save_predictions_json(str(out / "detections.json"), payload)
    save_predictions_csv(str(out / "detections.csv"), rows)
    return payload
