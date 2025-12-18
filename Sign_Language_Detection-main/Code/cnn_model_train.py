import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
import cv2
from glob import glob
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from tensorflow.keras import optimizers  # type: ignore[import]
from tensorflow.keras.models import Sequential  # type: ignore[import]
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # type: ignore[import]
from tensorflow.keras.utils import to_categorical  # type: ignore[import]
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore[import]
from tensorflow.keras import backend as K  # type: ignore[import]

# Get the project root directory (parent of Code folder)
PROJECT_ROOT = Path(__file__).parent.parent
GESTURES_DIR = PROJECT_ROOT / "gestures"

def get_image_size():
	# Find any gesture image to determine size
	gesture_folders = list(GESTURES_DIR.glob('*/'))
	if not gesture_folders:
		raise FileNotFoundError(f"No gesture folders found in '{GESTURES_DIR}'. Capture gestures first.")
	
	for folder in gesture_folders:
		images = list(folder.glob('*.jpg'))
		if images:
			img = cv2.imread(str(images[0]), 0)
			if img is not None:
				return img.shape
	
	raise FileNotFoundError(f"No valid gesture images found in '{GESTURES_DIR}'. Capture gestures first.")

image_x, image_y = get_image_size()

def cnn_model(num_of_classes: int):
	model = Sequential()
	# Improved architecture with batch normalization and better filters
	model.add(Conv2D(32, (3,3), input_shape=(image_x, image_y, 1), activation='relu', padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
	model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_of_classes, activation='softmax'))
	
	# Use Adam optimizer instead of SGD for better convergence
	adam = optimizers.Adam(learning_rate=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	filepath = str(PROJECT_ROOT / "cnn_model_keras2.h5")
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	return model, callbacks_list

def train():
	train_images_path = PROJECT_ROOT / "train_images"
	train_labels_path = PROJECT_ROOT / "train_labels"
	val_images_path = PROJECT_ROOT / "val_images"
	val_labels_path = PROJECT_ROOT / "val_labels"
	
	with open(train_images_path, "rb") as f:
		train_images = np.array(pickle.load(f))
	with open(train_labels_path, "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open(val_images_path, "rb") as f:
		val_images = np.array(pickle.load(f))
	with open(val_labels_path, "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	all_labels = np.concatenate((train_labels, val_labels))
	unique_labels = np.unique(all_labels)
	label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
	train_labels = np.vectorize(label_to_index.get)(train_labels)
	val_labels = np.vectorize(label_to_index.get)(val_labels)
	num_classes = len(unique_labels)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
	
	# Normalize pixel values to 0-1 range for better training
	train_images = train_images.astype('float32') / 255.0
	val_images = val_images.astype('float32') / 255.0
	
	train_labels = to_categorical(train_labels, num_classes=num_classes)
	val_labels = to_categorical(val_labels, num_classes=num_classes)

	print(val_labels.shape)

	model, callbacks_list = cnn_model(num_classes)
	model.summary()
	model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=64, callbacks=callbacks_list)
	scores = model.evaluate(val_images, val_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	print("CNN Accuracy: %.2f%%" % (scores[1]*100))

train()
K.clear_session();
