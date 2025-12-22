import os
import pickle
from pathlib import Path
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
GESTURES_DIR = PROJECT_ROOT / "gestures"


def get_image_size() -> Tuple[int, int]:
    gesture_folders = list(GESTURES_DIR.glob('*/'))
    if not gesture_folders:
        raise FileNotFoundError(f"No gesture folders found in '{GESTURES_DIR}'. Capture gestures first.")
    for folder in gesture_folders:
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        if images:
            img = cv2.imread(str(images[0]), 0)
            if img is not None:
                return img.shape
    raise FileNotFoundError(f"No valid gesture images found in '{GESTURES_DIR}'. Capture gestures first.")


def load_images_and_labels(image_x: int, image_y: int) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[str] = []

    class_folders = sorted([p for p in GESTURES_DIR.iterdir() if p.is_dir()])
    if not class_folders:
        raise FileNotFoundError(f"No class folders found in '{GESTURES_DIR}'.")

    for folder in class_folders:
        label = folder.name
        image_paths = sorted(glob(str(folder / '*.jpg'))) + sorted(glob(str(folder / '*.png')))
        for ip in image_paths:
            img = cv2.imread(ip, 0)
            if img is None:
                continue
            if img.shape != (image_x, image_y):
                img = cv2.resize(img, (image_y, image_x))
            X.append(img)
            y.append(label)

    if not X:
        raise ValueError("No images loaded from gestures directory.")

    return np.array(X), np.array(y)


def main():
    image_x, image_y = get_image_size()

    X, y = load_images_and_labels(image_x, image_y)

    # First split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Second split: val vs test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Save pickles at project root to match training script expectations
    with open(PROJECT_ROOT / "train_images", "wb") as f:
        pickle.dump(X_train, f)
    with open(PROJECT_ROOT / "train_labels", "wb") as f:
        pickle.dump(y_train, f)

    with open(PROJECT_ROOT / "val_images", "wb") as f:
        pickle.dump(X_val, f)
    with open(PROJECT_ROOT / "val_labels", "wb") as f:
        pickle.dump(y_val, f)

    with open(PROJECT_ROOT / "test_images", "wb") as f:
        pickle.dump(X_test, f)
    with open(PROJECT_ROOT / "test_labels", "wb") as f:
        pickle.dump(y_test, f)

    print("Dataset split complete:")
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")


if __name__ == "__main__":
    main()
