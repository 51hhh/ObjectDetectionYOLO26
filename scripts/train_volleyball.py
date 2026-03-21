from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "train_volleyball.yaml"


def resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    model_path = resolve_path(cfg.pop("model"))
    cfg["data"] = resolve_path(cfg["data"])
    cfg["project"] = resolve_path(cfg["project"])

    print("Training config:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    model = YOLO(model_path)
    model.train(**cfg)


if __name__ == "__main__":
    main()
