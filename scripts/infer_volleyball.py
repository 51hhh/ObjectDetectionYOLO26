from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urlparse

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weight" / "best.pt"
DEFAULT_PROJECT = ROOT / "runs"


def resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("source", help="Image, video, folder, webcam index, or stream URL")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to model weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.15, help="NMS IoU threshold")
    parser.add_argument("--device", default="cpu", help="CUDA device, e.g. 0 or cpu")
    parser.add_argument("--name", default="volleyball_predict", help="Output run name")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence in txt output")
    parser.add_argument("--show", action="store_true", help="Show prediction window")
    return parser.parse_args()


def is_stream_url(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def normalize_source(value: str):
    if value.isdigit():
        return int(value)
    if is_stream_url(value):
        return value
    return resolve_path(value)


def main():
    args = parse_args()
    weights = resolve_path(args.weights)
    source = normalize_source(args.source)
    project = str(DEFAULT_PROJECT.resolve())

    print("Inference config:")
    print(f"  weights: {weights}")
    print(f"  source: {source}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  conf: {args.conf}")
    print(f"  iou: {args.iou}")
    print(f"  device: {args.device}")
    print(f"  output: {project}/{args.name}")

    model = YOLO(weights)
    model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
    )


if __name__ == "__main__":
    main()
