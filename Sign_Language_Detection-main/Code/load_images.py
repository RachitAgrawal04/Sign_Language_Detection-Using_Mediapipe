from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.utils import shuffle


BASE_DIR = Path(__file__).resolve().parents[1]
GESTURES_DIR = BASE_DIR / "gestures"
OUTPUT_FILES = {
	"train_images": BASE_DIR / "train_images",
	"train_labels": BASE_DIR / "train_labels",
	"test_images": BASE_DIR / "test_images",
	"test_labels": BASE_DIR / "test_labels",
	"val_images": BASE_DIR / "val_images",
	"val_labels": BASE_DIR / "val_labels",
}


def load_images_and_labels() -> list[tuple[np.ndarray, int]]:
	if not GESTURES_DIR.exists():
		raise FileNotFoundError(
			f"Gestures directory not found at {GESTURES_DIR}. Capture gestures first."
		)

	image_paths = sorted(GESTURES_DIR.glob("*/*.jpg"))
	if not image_paths:
		raise FileNotFoundError(
			f"No gesture images found in {GESTURES_DIR}. Capture gestures before splitting."
		)

	dataset: list[tuple[np.ndarray, int]] = []
	for img_path in image_paths:
		label_str = img_path.parent.name
		if not label_str.isdigit():
			print(f"⚠ Skipping {img_path}: folder name is not a digit.")
			continue

		image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
		if image is None:
			print(f"⚠ Unable to read image {img_path}, skipping.")
			continue

		dataset.append((image.astype(np.uint8), int(label_str)))

	if not dataset:
		raise RuntimeError("All images failed to load. Check the dataset integrity.")

	return dataset


def save_pickle(path: Path, data) -> None:
	with path.open("wb") as f:
		pickle.dump(data, f)


def main() -> None:
	print("Loading gesture images from:", GESTURES_DIR)
	images_labels = load_images_and_labels()
	print(f"Loaded {len(images_labels)} labeled images")

	images_labels = shuffle(images_labels, random_state=42)
	images, labels = zip(*images_labels)

	total = len(images)
	train_end = int(5 / 6 * total)
	val_start = int(11 / 12 * total)

	train_images = images[:train_end]
	train_labels = labels[:train_end]
	test_images = images[train_end:val_start]
	test_labels = labels[train_end:val_start]
	val_images = images[val_start:]
	val_labels = labels[val_start:]

	splits = {
		"train_images": train_images,
		"train_labels": train_labels,
		"test_images": test_images,
		"test_labels": test_labels,
		"val_images": val_images,
		"val_labels": val_labels,
	}

	for name, data in splits.items():
		save_pickle(OUTPUT_FILES[name], data)
		print(f"Saved {name}: {len(data)} entries")


if __name__ == "__main__":
	main()
