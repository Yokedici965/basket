from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BASE / "data" / "hoops" / "dataset" / "dataset.yaml"
MODELS_DIR = BASE / "models"
OUTPUT_DIR = BASE / "runs" / "hoop_training"


def train(
    data: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    name: str,
) -> None:
    data = data.resolve()
    if not data.exists():
        raise FileNotFoundError(data)

    model_path = Path(base_model)
    if not model_path.exists():
        model_local = MODELS_DIR / base_model
        if model_local.exists():
            model_path = model_local
        else:
            model_path = Path(base_model)
    model = YOLO(str(model_path))

    project.mkdir(parents=True, exist_ok=True)
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    print("[TRAIN] finished", results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hoop detector with Ultralytics YOLO")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="dataset yaml path")
    parser.add_argument("--model", default="yolov8n.pt", help="base model weight")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="device (e.g. 0 or cpu)")
    parser.add_argument("--name", default="exp", help="run name")
    args = parser.parse_args()

    train(
        data=Path(args.data),
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=OUTPUT_DIR,
        name=args.name,
    )


if __name__ == "__main__":
    main()
