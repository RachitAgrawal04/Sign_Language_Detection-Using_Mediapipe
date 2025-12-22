import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent

def evaluate():
	# Load test data
	with open(PROJECT_ROOT / "test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open(PROJECT_ROOT / "test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f))

	# Load label mapping
	label_map_path = PROJECT_ROOT / "label_map.pkl"
	if not label_map_path.exists():
		raise FileNotFoundError("label_map.pkl not found. Train the model first.")
	with open(label_map_path, "rb") as lm:
		label_to_index = pickle.load(lm)

	# Build index to label mapping for reporting
	index_to_label = {v: k for k, v in label_to_index.items()}

	# Preprocess
	X = test_images.reshape(-1, test_images.shape[1], test_images.shape[2], 1).astype('float32') / 255.0
	y_true = np.array([label_to_index[lbl] for lbl in test_labels])

	# Load model and predict
	model = load_model(PROJECT_ROOT / "cnn_model_keras2.keras")
	y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

	# Print metrics
	print("\n" + "="*70)
	print("TEST SET METRICS")
	print("="*70)
	print(classification_report(y_true, y_pred, target_names=[index_to_label[i] for i in sorted(index_to_label.keys())]))

	# Confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	print(f"\nConfusion Matrix shape: {cm.shape}")
	print(f"Test Accuracy: {(y_pred == y_true).sum() / len(y_true) * 100:.2f}%")

if __name__ == "__main__":
	evaluate()
