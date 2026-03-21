import io
import json
import random
import shutil
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SETTINGS = {
    "positive_zip": "images.zip",
    "negative_zip": "negative_samples.zip",
    "output_dir": "coco",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
}


@dataclass
class Sample:
    source: str
    image_member: str
    width: int
    height: int
    objects: list[tuple[str, float, float, float, float]]


def parse_xml_bytes(data: bytes):
    root = ET.fromstring(data)
    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    width = int(float(size.findtext("width", default="0"))) if size is not None else 0
    height = int(float(size.findtext("height", default="0"))) if size is not None else 0
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))
        objects.append((name, xmin, ymin, xmax, ymax))
    return filename, width, height, objects


def parse_labelme_bytes(data: bytes):
    payload = json.loads(data)
    filename = (payload.get("imagePath") or "").strip()
    width = int(payload.get("imageWidth") or 0)
    height = int(payload.get("imageHeight") or 0)
    objects = []
    for shape in payload.get("shapes", []):
        label = (shape.get("label") or "").strip()
        points = shape.get("points") or []
        if not label or len(points) < 2:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        objects.append((label, min(xs), min(ys), max(xs), max(ys)))
    return filename, width, height, objects


def clamp_box(box, width: int, height: int):
    name, xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(xmin, width - 1))
    ymin = max(0.0, min(ymin, height - 1))
    xmax = max(0.0, min(xmax, width))
    ymax = max(0.0, min(ymax, height))
    if xmax <= xmin or ymax <= ymin:
        return None
    return name, xmin, ymin, xmax, ymax


def get_image_size(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as src:
        data = src.read()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
    return int(width), int(height)


def load_positive_samples(zip_path: Path):
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        names = {n for n in zf.namelist() if not n.endswith("/")}
        image_members = sorted(
            n
            for n in names
            if Path(n).suffix.lower() in IMG_EXTS and Path(n).parts and Path(n).parts[0] == "images"
        )
        if not image_members:
            raise FileNotFoundError(f"No images found in {zip_path}")

        for image_member in image_members:
            stem = Path(image_member).stem
            xml_member = f"annotations/{stem}.xml"
            json_member = f"images/{stem}.json"
            filename = ""
            width = 0
            height = 0
            objects = []

            if xml_member in names:
                filename, width, height, objects = parse_xml_bytes(zf.read(xml_member))
            elif json_member in names:
                filename, width, height, objects = parse_labelme_bytes(zf.read(json_member))

            if (width <= 0 or height <= 0) and json_member in names:
                j_filename, j_width, j_height, j_objects = parse_labelme_bytes(zf.read(json_member))
                if not filename:
                    filename = j_filename
                if width <= 0 or height <= 0:
                    width, height = j_width, j_height
                if not objects:
                    objects = j_objects

            if width <= 0 or height <= 0:
                width, height = get_image_size(zf, image_member)

            if width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid image size for {image_member}")

            clamped_objects = []
            for obj in objects:
                fixed = clamp_box(obj, width, height)
                if fixed is not None:
                    clamped_objects.append(fixed)

            samples.append(Sample(source="positive", image_member=image_member, width=width, height=height, objects=clamped_objects))
    return samples


def load_negative_samples(zip_path: Path):
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        image_members = sorted(
            n for n in zf.namelist() if not n.endswith("/") and Path(n).suffix.lower() in IMG_EXTS
        )
        if not image_members:
            raise FileNotFoundError(f"No images found in {zip_path}")

        for image_member in image_members:
            width, height = get_image_size(zf, image_member)
            samples.append(Sample(source="negative", image_member=image_member, width=width, height=height, objects=[]))
    return samples


def split_bucket(items: list[Sample]):
    n = len(items)
    n_train = int(n * SETTINGS["train_ratio"])
    n_val = int(n * SETTINGS["val_ratio"])
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def split_samples(samples: list[Sample], seed: int):
    positives = [s for s in samples if s.objects]
    negatives = [s for s in samples if not s.objects]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positive_splits = split_bucket(positives)
    negative_splits = split_bucket(negatives)
    merged = {}
    for split in SPLITS:
        part = positive_splits[split] + negative_splits[split]
        rng.shuffle(part)
        merged[split] = part
    return merged


def ensure_output_dir(out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already exists and is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)


def unique_output_name(base_name: str, used_names: set[str]):
    name = Path(base_name).name
    if name not in used_names:
        used_names.add(name)
        return name

    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def extract_image(zf: zipfile.ZipFile, member: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yolo_label(path: Path, sample: Sample, cls_to_yolo: dict[str, int]):
    lines = []
    for cls_name, xmin, ymin, xmax, ymax in sample.objects:
        cx = ((xmin + xmax) / 2.0) / sample.width
        cy = ((ymin + ymax) / 2.0) / sample.height
        w = (xmax - xmin) / sample.width
        h = (ymax - ymin) / sample.height
        lines.append(f"{cls_to_yolo[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dump_text(path, lines)


def build_categories(class_names: list[str]):
    return [{"id": idx, "name": name, "supercategory": "object"} for idx, name in enumerate(class_names, start=1)]


def write_data_yaml(path: Path, dataset_root: Path, class_names: list[str]):
    lines = [
        f'path: "{dataset_root.as_posix()}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    total_ratio = SETTINGS["train_ratio"] + SETTINGS["val_ratio"] + SETTINGS["test_ratio"]
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    root = Path(__file__).resolve().parents[1]
    positive_zip = root / SETTINGS["positive_zip"]
    negative_zip = root / SETTINGS["negative_zip"]
    out_dir = root / SETTINGS["output_dir"]

    if not positive_zip.exists():
        raise FileNotFoundError(f"Missing {positive_zip}")
    if not negative_zip.exists():
        raise FileNotFoundError(f"Missing {negative_zip}")

    ensure_output_dir(out_dir)

    positive_samples = load_positive_samples(positive_zip)
    negative_samples = load_negative_samples(negative_zip)
    all_samples = positive_samples + negative_samples

    class_names = sorted({obj[0] for sample in positive_samples for obj in sample.objects})
    if not class_names:
        raise RuntimeError("No classes found in positive annotations")

    cls_to_yolo = {name: idx for idx, name in enumerate(class_names)}
    cls_to_coco = {name: idx for idx, name in enumerate(class_names, start=1)}
    categories = build_categories(class_names)
    split_map = split_samples(all_samples, SETTINGS["seed"])

    split_summary = {}
    zip_handles = {
        "positive": zipfile.ZipFile(positive_zip),
        "negative": zipfile.ZipFile(negative_zip),
    }

    try:
        for split in SPLITS:
            samples = split_map[split]
            used_names = set()
            image_records = []
            annotation_records = []
            image_lines = []
            ann_id = 1

            for img_id, sample in enumerate(samples, start=1):
                output_name = unique_output_name(Path(sample.image_member).name, used_names)
                image_rel = Path("images") / split / output_name
                label_rel = Path("labels") / split / f"{Path(output_name).stem}.txt"
                extract_image(zip_handles[sample.source], sample.image_member, out_dir / image_rel)
                write_yolo_label(out_dir / label_rel, sample, cls_to_yolo)

                image_records.append({
                    "id": img_id,
                    "file_name": output_name,
                    "width": sample.width,
                    "height": sample.height,
                })
                image_lines.append(image_rel.as_posix())

                for cls_name, xmin, ymin, xmax, ymax in sample.objects:
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    annotation_records.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_to_coco[cls_name],
                        "bbox": [round(xmin, 3), round(ymin, 3), round(box_w, 3), round(box_h, 3)],
                        "area": round(box_w * box_h, 3),
                        "iscrowd": 0,
                    })
                    ann_id += 1

            dump_json(
                out_dir / "annotations" / f"instances_{split}.json",
                {
                    "images": image_records,
                    "annotations": annotation_records,
                    "categories": categories,
                },
            )
            dump_text(out_dir / f"{split}.txt", image_lines)
            split_summary[split] = {
                "images": len(samples),
                "positive_images": sum(1 for s in samples if s.objects),
                "negative_images": sum(1 for s in samples if not s.objects),
                "annotations": len(annotation_records),
            }
    finally:
        for zf in zip_handles.values():
            zf.close()

    write_data_yaml(out_dir / "data.yaml", out_dir, class_names)
    dump_json(
        out_dir / "summary.json",
        {
            "output_dir": out_dir.as_posix(),
            "positive_zip": positive_zip.as_posix(),
            "negative_zip": negative_zip.as_posix(),
            "classes": class_names,
            "num_classes": len(class_names),
            "images_total": len(all_samples),
            "positive_images": sum(1 for s in all_samples if s.objects),
            "negative_images": sum(1 for s in all_samples if not s.objects),
            "annotations_total": sum(len(s.objects) for s in all_samples),
            "splits": split_summary,
        },
    )

    print(json.dumps({
        "output_dir": out_dir.as_posix(),
        "classes": class_names,
        "splits": split_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
