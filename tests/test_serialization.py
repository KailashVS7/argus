import csv
import json

from argus.serialization import save_predictions_csv, save_predictions_json


def test_prediction_serialization(tmp_path):
    payload = {
        "source_image": "x.tif",
        "detections": [{"class": "vessel", "confidence": 0.9}],
        "preprocessing": {"input_mode": "separate"},
    }
    rows = [
        {
            "source_image": "x.tif",
            "class": "vessel",
            "confidence": 0.9,
            "xmin": 1,
            "ymin": 2,
            "xmax": 3,
            "ymax": 4,
        }
    ]

    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"
    save_predictions_json(str(json_path), payload)
    save_predictions_csv(str(csv_path), rows)

    assert json.loads(json_path.read_text(encoding="utf-8"))["source_image"] == "x.tif"
    with csv_path.open(encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 1
    assert records[0]["class"] == "vessel"
