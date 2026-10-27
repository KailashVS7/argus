import json
import subprocess
import sys


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "argus.cli", *args], check=True, capture_output=True, text=True)


def test_cli_smoke_demo_and_infer(tmp_path):
    demo_dir = tmp_path / "demo"
    out_dir = tmp_path / "outputs"

    run_cli("make-demo", "--output-dir", str(demo_dir))
    assert (demo_dir / "synthetic_vv.tif").exists()
    assert (demo_dir / "synthetic_vh.tif").exists()

    run_cli(
        "infer",
        "--vv",
        str(demo_dir / "synthetic_vv.tif"),
        "--vh",
        str(demo_dir / "synthetic_vh.tif"),
        "--weights",
        "none",
        "--no-download",
        "--output-dir",
        str(out_dir),
        "--score-threshold",
        "1.1",
    )

    detections = json.loads((out_dir / "detections.json").read_text(encoding="utf-8"))
    assert detections["preprocessing"]["channel_order"] == ["vv", "vh", "vv_minus_vh"]
    assert "not proof of ghost vessels" in " ".join(detections["notes"]).lower()
    assert (out_dir / "annotated.png").exists()
    assert (out_dir / "detections.csv").exists()
