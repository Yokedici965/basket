from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import shutil

BASE = Path(__file__).resolve().parents[1]
LABELS_ROOT = BASE / "data" / "hoops" / "labels_v1"
SAMPLES_ROOT = BASE / "configs" / "calibrations"
DATASET_ROOT = BASE / "data" / "hoops" / "dataset"


def load_labels() -> List[Dict[str, object]]:
    labels: List[Dict[str, object]] = []
    for video_dir in LABELS_ROOT.iterdir():
        if not video_dir.is_dir():
            continue
        for label_path in sorted(video_dir.glob("*.json")):
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            payload["label_path"] = label_path
            labels.append(payload)
    if not labels:
        raise RuntimeError(f"No labels found in {LABELS_ROOT}")
    return labels


def resolve_raw_image(item: Dict[str, object]) -> Path:
    video = Path(str(item["video"]))
    stem = video.stem
    frame = int(item["frame"])
    ts = float(item.get("ts", 0.0))
    raw_dir = SAMPLES_ROOT / f"{stem}_samples" / "raw"
    pattern = f"frame_{frame:06d}_t{ts:.2f}.jpg"
    path = raw_dir / pattern
    if path.exists():
        return path
    candidates = list(raw_dir.glob(f"frame_{frame:06d}_t*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"Raw sample not found for {pattern}")
    def diff(p: Path) -> float:
        part = p.stem.split('_t')[-1]
        try:
            return abs(float(part) - ts)
        except ValueError:
            return float('inf')
    candidates.sort(key=diff)
    return candidates[0]


def load_meta(item: Dict[str, object]) -> Dict[str, object]:
    stem = Path(str(item["video"])).stem
    meta_path = SAMPLES_ROOT / f"{stem}_samples" / "meta" / f"frame_{int(item['frame']):06d}.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def convert_to_yolo(item: Dict[str, object], width: int, height: int) -> List[str]:
    lines: List[str] = []
    labels = item.get("labels", {})
    for hoop in labels.values():
        cx = float(hoop["x"]) / width
        cy = float(hoop["y"]) / height
        radius = float(hoop["radius"])
        w = (radius * 2) / width
        h = (radius * 2) / height
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def prepare_dataset(train_split: float = 0.8, seed: int = 42) -> None:
    labels = load_labels()
    random.Random(seed).shuffle(labels)

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    images_train = DATASET_ROOT / "images" / "train"
    images_val = DATASET_ROOT / "images" / "val"
    labels_train = DATASET_ROOT / "labels" / "train"
    labels_val = DATASET_ROOT / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    split_idx = max(1, int(len(labels) * train_split))
    train_items = labels[:split_idx]
    val_items = labels[split_idx:] or labels[: max(1, len(labels) // 5)]

    def process(items: List[Dict[str, object]], img_dir: Path, lbl_dir: Path):
        for item in items:
            try:
                raw_path = resolve_raw_image(item)
                meta = load_meta(item)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}")
                continue
            width = int(meta.get("width", 1920))
            height = int(meta.get("height", 1080))
            yolo_lines = convert_to_yolo(item, width, height)
            if not yolo_lines:
                continue
            stem = f"{Path(str(item['video'])).stem}_{int(item['frame']):06d}"
            dst_img = img_dir / f"{stem}.jpg"
            shutil.copy(raw_path, dst_img)
            dst_lbl = lbl_dir / f"{stem}.txt"
            dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")

    process(train_items, images_train, labels_train)
    process(val_items, images_val, labels_val)

    yaml_path = DATASET_ROOT / "dataset.yaml"
    yaml_content = (
        f"path: {DATASET_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['hoop']\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    summary = {
        "total_labels": len(labels),
        "train_images": len(list(images_train.glob('*.jpg'))),
        "val_images": len(list(images_val.glob('*.jpg'))),
    }
    print("[DATASET]", summary)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from hoop labels")
    parser.add_argument("--train-split", type=float, default=0.8)
    args = parser.parse_args()
    prepare_dataset(train_split=args.train_split)


if __name__ == "__main__":
    main()
