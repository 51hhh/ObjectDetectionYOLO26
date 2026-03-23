from argparse import ArgumentParser
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weight" / "best.pt"
DEFAULT_EXPORT_DIR = ROOT / "deploy" / "agx_zed" / "models"


def resolve_path(value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def default_output_path(weights_path: Path) -> Path:
    return DEFAULT_EXPORT_DIR / f"{weights_path.stem}.onnx"


def build_export_kwargs(imgsz=640, opset=17, dynamic=False, simplify=False, output_path=None):
    kwargs = {
        "format": "onnx",
        "imgsz": imgsz,
        "opset": opset,
        "dynamic": dynamic,
        "simplify": simplify,
    }
    if output_path is not None:
        kwargs["path"] = Path(output_path)
    return kwargs


def _load_yolo_class():
    from ultralytics import YOLO

    return YOLO


def export_model(weights_path: Path, output_path: Path, imgsz=640, opset=17, dynamic=False, simplify=False):
    YOLO = _load_yolo_class()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = build_export_kwargs(
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        output_path=output_path,
    )
    desired_path = Path(export_kwargs.pop("path", output_path))

    model = YOLO(str(weights_path))
    exported = model.export(**export_kwargs)

    exported_path = None
    if isinstance(exported, (str, Path)):
        exported_path = Path(exported)
    elif isinstance(exported, (list, tuple)) and exported:
        exported_path = Path(exported[0])

    if exported_path and exported_path.exists() and exported_path.resolve() != desired_path.resolve():
        desired_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), str(desired_path))
        return str(desired_path)

    return exported


def parse_args():
    parser = ArgumentParser(description="Export trained YOLO26 weights to ONNX for AGX deployment")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to .pt weights")
    parser.add_argument("--output", default=None, help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Static export image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes")
    parser.add_argument("--simplify", action="store_true", help="Enable ONNX simplification when onnxsim is available")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = resolve_path(args.weights)
    output_path = resolve_path(args.output) if args.output else default_output_path(weights_path)

    print("Export config:")
    print(f"  weights: {weights_path}")
    print(f"  output: {output_path}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  opset: {args.opset}")
    print(f"  dynamic: {args.dynamic}")
    print(f"  simplify: {args.simplify}")

    exported = export_model(
        weights_path=weights_path,
        output_path=output_path,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )

    print(f"Export complete: {exported}")


if __name__ == "__main__":
    main()
