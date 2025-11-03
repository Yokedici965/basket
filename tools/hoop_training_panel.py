from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def run_prepare_dataset() -> None:
    cmd = [str(PYTHON), str(BASE / "tools" / "prepare_hoop_dataset.py")]
    print("\n[Panel] Running dataset preparation...\n")
    subprocess.run(cmd, check=False)


def run_training() -> None:
    data_yaml = BASE / "data" / "hoops" / "dataset" / "dataset.yaml"
    if not data_yaml.exists():
        print("[Panel] Dataset YAML bulunamadı. Önce dataset hazırlayın.\n")
        return
    model = input(f"Base model (default yolov8n.pt): ") or "yolov8n.pt"
    epochs = input("Epoch sayısı (default 50): ") or "50"
    imgsz = input("Image size (default 640): ") or "640"
    batch = input("Batch size (default 16): ") or "16"
    device = input("Device (default 0 / cpu): ") or "0"
    name = input("Run name (default exp): ") or "exp"

    cmd = [
        str(PYTHON),
        str(BASE / "tools" / "train_hoop_model.py"),
        "--data",
        str(data_yaml),
        "--model",
        model,
        "--epochs",
        epochs,
        "--imgsz",
        imgsz,
        "--batch",
        batch,
        "--device",
        device,
        "--name",
        name,
    ]
    print("\n[Panel] Starting training...\n")
    subprocess.run(cmd, check=False)


def menu() -> None:
    while True:
        print("""
================ Hoop Training Panel ================
1) Dataset hazırla (labels_v1 -> YOLO format)
2) Model eğitimini başlat
3) Çıkış
""")
        choice = input("Seçim: ").strip()
        if choice == "1":
            run_prepare_dataset()
        elif choice == "2":
            run_training()
        elif choice == "3":
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçim\n")


def main() -> None:
    menu()


if __name__ == "__main__":
    main()
