import json

import numpy as np

from argus.preprocessing import build_three_channel, save_preprocessed


def test_channel_construction_and_normalization(tmp_path):
    vv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vh = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

    tensor, meta = build_three_channel(vv=vv, vh=vh, source="unit", input_mode="separate")

    assert tensor.shape == (3, 2, 2)
    assert np.isfinite(tensor).all()
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert meta.channel_order == ["vv", "vh", "vv_minus_vh"]

    out_npy = tmp_path / "tensor.npy"
    out_json = tmp_path / "meta.json"
    save_preprocessed(str(out_npy), str(out_json), tensor, meta)

    assert out_npy.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["input_mode"] == "separate"
    assert len(data["stats"]) == 3
