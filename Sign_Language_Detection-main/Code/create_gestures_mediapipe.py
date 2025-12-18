import os

# Silence verbose TensorFlow/MediaPipe logging before those libraries load
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Hide INFO and WARNING logs
os.environ.setdefault("GLOG_minloglevel", "2")      # Suppress MediaPipe's glog warnings
os.environ.setdefault("GLOG_logtostderr", "1")

import cv2
import numpy as np
import sqlite3
import sys
import contextlib
from absl import logging as absl_logging
import mediapipe as mp
from pathlib import Path

# Route MediaPipe/absl logs to error level only
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.use_absl_handler()

# MediaPipe hand detection
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]

image_x, image_y = 50, 50

BASE_DIR = Path(__file__).resolve().parents[1]
GESTURES_DIR = BASE_DIR / "gestures"
DB_PATH = BASE_DIR / "gesture_db.db"


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Temporarily redirect stdout and stderr to keep console clean."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


@contextlib.contextmanager
def suppress_native_stderr():
    """Redirect the C-level stderr file descriptor to /dev/null temporarily."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)

def init_create_folder_database():
    if not GESTURES_DIR.exists():
        GESTURES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created gestures directory: {GESTURES_DIR}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gesture'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()
        print(f"✓ Created database table: {DB_PATH}")
    
    conn.close()

def store_in_db(g_id, g_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Try to insert new gesture
        cursor.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (g_id, g_name))
        conn.commit()
        print(f"✓ Gesture {g_id} → '{g_name}' saved to database")
    except sqlite3.IntegrityError:
        # Gesture ID already exists
        existing = cursor.execute("SELECT g_name FROM gesture WHERE g_id = ?", (g_id,)).fetchone()
        print(f"\n⚠ Gesture ID {g_id} already exists with name: '{existing[0]}'")
        choice = input(f"Do you want to update it to '{g_name}'? (y/n): ")
        
        if choice.lower() == 'y':
            cursor.execute("UPDATE gesture SET g_name = ? WHERE g_id = ?", (g_name, g_id))
            conn.commit()
            print(f"✓ Updated gesture {g_id} → '{g_name}'")
        else:
            print("Keeping existing gesture. No changes made.")
    finally:
        conn.close()

def open_camera(indices=(0, 1)):
    for idx in indices:
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            ret, _ = cam.read()
            if ret:
                if idx != indices[0]:
                    print(f"Using fallback camera index {idx}")
                return cam
        cam.release()
    raise RuntimeError("Could not access any camera. Check connections and retry.")

def extract_hand_region(image, hand_landmarks):
    """Extract hand region from image using MediaPipe landmarks"""
    h, w, _ = image.shape
    
    # Get bounding box from landmarks
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
    
    x_min = int(min(x_coords) * w)
    x_max = int(max(x_coords) * w)
    y_min = int(min(y_coords) * h)
    y_max = int(max(y_coords) * h)
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    
    # Extract and convert to grayscale
    hand_img = image[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Make square
    h_crop, w_crop = binary.shape
    if w_crop > h_crop:
        diff = w_crop - h_crop
        top = diff // 2
        bottom = diff - top
        binary = cv2.copyMakeBorder(binary, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,))
    elif h_crop > w_crop:
        diff = h_crop - w_crop
        left = diff // 2
        right = diff - left
        binary = cv2.copyMakeBorder(binary, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,))
    
    # Resize to target size
    binary = cv2.resize(binary, (image_x, image_y))
    
    return binary

def store_images(g_id):
    total_pics = 1200
    cam = open_camera()

    with suppress_native_stderr(), suppress_stdout_stderr():
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    gesture_dir = GESTURES_DIR / str(g_id)
    gesture_dir.mkdir(parents=True, exist_ok=True)
    
    pic_no = 0
    flag_start_capturing = False
    
    print("\n" + "="*60)
    print("  CAPTURE INSTRUCTIONS")
    print("="*60)
    print("  - Position your hand in camera view")
    print("  - Wait for 'Hand Detected' (green text)")
    print("  - Press 'c' to START capturing")
    print("  - Hold gesture and move hand slightly (different angles)")
    print("  - Press 'c' again to PAUSE")
    print("  - Press 'q' to QUIT")
    print(f"  - Goal: {total_pics} images for gesture {g_id}")
    print("="*60 + "\n")
    
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe (suppress verbose logging)
            with suppress_native_stderr(), suppress_stdout_stderr():
                results = hands.process(img_rgb)

            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )

                    # Capture if flag is set
                    if flag_start_capturing:
                        try:
                            hand_crop = extract_hand_region(img, hand_landmarks)
                            pic_no += 1
                            save_path = gesture_dir / f"{pic_no}.jpg"
                            cv2.imwrite(str(save_path), hand_crop)

                            # Show capturing indicator
                            cv2.putText(img, "CAPTURING...", (30, 60),
                                       cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3)
                        except Exception as e:
                            print(f"⚠ Error capturing frame {pic_no}: {e}")

            # Display status
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            status_text = "Hand Detected ✓" if hand_detected else "No Hand ✗"
            cv2.putText(img, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # Progress counter
            progress_text = f"Images: {pic_no}/{total_pics}"
            progress_color = (0, 255, 0) if pic_no < total_pics else (255, 0, 255)
            cv2.putText(img, progress_text, (30, 100),
                       cv2.FONT_HERSHEY_TRIPLEX, 1.5, progress_color, 2)

            # Recording indicator (red circle)
            if flag_start_capturing:
                cv2.circle(img, (img.shape[1] - 30, 30), 15, (0, 0, 255), -1)
                cv2.putText(img, "REC", (img.shape[1] - 70, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow(f"Capturing Gesture {g_id} - MediaPipe", img)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('c'):
                flag_start_capturing = not flag_start_capturing
                if flag_start_capturing:
                    print(f"▶ Started capturing gesture {g_id}...")
                else:
                    print(f"⏸ Paused at {pic_no} images")
            elif keypress == ord('q'):
                print(f"\n⚠ Quit requested. Captured {pic_no}/{total_pics} images.")
                break

            if pic_no >= total_pics:
                print(f"\n✓ SUCCESS! Captured all {pic_no} images for gesture {g_id}!")
                print(f"✓ Saved to: {gesture_dir}")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        hands.close()

def prompt_gesture_details():
    """Prompt the user for gesture ID and name with validation."""
    print("\n" + "=" * 50)
    print("  GESTURE CAPTURE - MediaPipe")
    print("=" * 50)
    print("\nExample: If capturing letter 'A', enter:")
    print("  Gesture ID: 1")
    print("  Gesture Name: A")
    print("-" * 50)

    while True:
        g_id_input = input("\nEnter gesture ID (number): ").strip()
        if not g_id_input:
            print("⚠ Gesture ID cannot be empty. Please enter a number like 1, 2, 3...")
            continue
        if not g_id_input.isdigit():
            print("⚠ Gesture ID must be a whole number. Try again.")
            continue
        g_id = int(g_id_input)
        break

    while True:
        g_name = input("Enter gesture name (letter/word): ").strip()
        if not g_name:
            print("⚠ Gesture name cannot be empty. Please enter a letter or word.")
            continue
        break

    print(f"\nYou entered: ID={g_id}, Name={g_name}")
    confirm = input("Is this correct? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Let's try again...")
        return prompt_gesture_details()

    return g_id, g_name


if __name__ == "__main__":
    init_create_folder_database()
    gesture_id, gesture_name = prompt_gesture_details()
    store_in_db(gesture_id, gesture_name)
    store_images(gesture_id)
