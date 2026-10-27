from __future__ import annotations

import argparse

from .demo import generate_synthetic_demo
from .inference import run_inference
from .modeling import build_model, load_checkpoint, save_checkpoint
from .preprocessing import load_vv_vh, save_preprocessed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argus", description="Argus Proto 1 (offline SAR vessel detection baseline)")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("make-demo", help="Generate deterministic synthetic demo inputs")
    demo.add_argument("--output-dir", required=True)

    prep = sub.add_parser("preprocess", help="Create VV/VH/derived tensor")
    prep.add_argument("--vv")
    prep.add_argument("--vh")
    prep.add_argument("--input")
    prep.add_argument("--output-npy", required=True)
    prep.add_argument("--output-json", required=True)

    init = sub.add_parser("init-model", help="Initialize VGG16+FasterRCNN and optionally save checkpoint")
    init.add_argument("--weights", choices=["none", "imagenet"], default="none")
    init.add_argument("--no-download", action="store_true")
    init.add_argument("--checkpoint", required=True)

    infer = sub.add_parser("infer", help="Run CPU inference and serialize outputs")
    infer.add_argument("--vv")
    infer.add_argument("--vh")
    infer.add_argument("--input")
    infer.add_argument("--weights", choices=["none", "imagenet"], default="none")
    infer.add_argument("--no-download", action="store_true")
    infer.add_argument("--checkpoint")
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--score-threshold", type=float, default=0.5)

    sub.add_parser("gui", help="Launch lightweight local GUI (tkinter)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "make-demo":
        created = generate_synthetic_demo(args.output_dir)
        for key, value in created.items():
            print(f"{key}: {value}")
        return 0

    if args.command == "preprocess":
        tensor, metadata = load_vv_vh(vv_path=args.vv, vh_path=args.vh, input_path=args.input)
        save_preprocessed(args.output_npy, args.output_json, tensor, metadata)
        print(f"saved tensor={args.output_npy} metadata={args.output_json}")
        return 0

    if args.command == "init-model":
        model, info = build_model(weights=args.weights, no_download=args.no_download)
        save_checkpoint(args.checkpoint, model, info)
        print(f"saved checkpoint={args.checkpoint} imagenet_used={info.used_imagenet_weights}")
        return 0

    if args.command == "infer":
        model, info = build_model(weights=args.weights, no_download=args.no_download)
        if args.checkpoint:
            load_meta = load_checkpoint(args.checkpoint, model, strict=False)
            print(f"loaded checkpoint metadata={load_meta}")

        payload = run_inference(
            model=model,
            vv_path=args.vv,
            vh_path=args.vh,
            input_path=args.input,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
        )
        print(f"saved outputs to {args.output_dir}; detections={len(payload['detections'])}; imagenet_used={info.used_imagenet_weights}")
        return 0

    if args.command == "gui":
        from .gui import launch_gui

        launch_gui()
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
