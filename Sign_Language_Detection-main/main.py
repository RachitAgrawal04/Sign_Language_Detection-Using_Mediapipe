import cv2
import time
import os
import numpy as np
import HandTrackingModule as htm

# Configuration
hCam, wCam = 720, 1280
CONFIDENCE_THRESHOLD = 0.7

# Colors (BGR format)
COLOR_PRIMARY = (46, 204, 113)      # Green
COLOR_SECONDARY = (52, 152, 219)    # Blue
COLOR_ACCENT = (231, 76, 60)        # Red
COLOR_TEXT = (255, 255, 255)        # White
COLOR_BG = (44, 62, 80)             # Dark Blue-Gray
COLOR_PANEL = (52, 73, 94)          # Panel Gray

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

detector = htm.handDetector(detectionCon=0.5, trackCon=0.5)

# History tracking
detected_history = []
MAX_HISTORY = 10
confidence_scores = []
fps_list = []
typed_text = ""
INSTRUCTION_LINES = [
    "1. Keep your hand centered in frame",
    "2. Hold each sign steady for a moment",
    "3. Press 'Space' to insert blank",
    "4. Press 'C' to clear history",
    "5. Press 'Q' to exit",
]
LEGEND_ROWS = [
    "A  B  C  D  E",
    "F  G  H  I  K",
    "L  M  N  O  P",
    "Q  R  S  T  U",
    "V  W  X  Y",
]

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=20):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw rectangles
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw circles at corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    
    
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

def draw_ui_panel(img):
    """Draw the main UI panel"""
    panel_height = 200
    panel_y = img.shape[0] - panel_height
    
   
    overlay = img.copy()
    cv2.rectangle(overlay, (0, panel_y), (img.shape[1], img.shape[0]), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    

    cv2.line(img, (0, panel_y), (img.shape[1], panel_y), COLOR_PRIMARY, 3)
    
    return panel_y

def draw_header(img):
    """Draw header with title"""
  
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
 
    cv2.putText(img, "ASL Sign Language Detection", (20, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOR_PRIMARY, 2)
  
    cv2.line(img, (0, 80), (img.shape[1], 80), COLOR_PRIMARY, 3)

def draw_detected_sign(img, result, x, y):
    """Draw the detected sign with enhanced styling"""
    if result:
        # Main sign box
        box_width = 300
        box_height = 200
        box_x = x - box_width // 2
        box_y = y
        
        # Background
        overlay = img.copy()
        draw_rounded_rectangle(overlay, (box_x, box_y), 
                             (box_x + box_width, box_y + box_height), 
                             COLOR_PRIMARY, -1, 30)
        cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
        
      
        draw_rounded_rectangle(img, (box_x, box_y), 
                             (box_x + box_width, box_y + box_height), 
                             COLOR_PRIMARY, 4, 30)
        
        # Label
        cv2.putText(img, "Detected Sign:", (box_x + 20, box_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        # Sign letter
        cv2.putText(img, result, (box_x + box_width//2 - 40, box_y + 140), 
                   cv2.FONT_HERSHEY_DUPLEX, 4, COLOR_TEXT, 6)
    else:
        # No sign detected message
        cv2.putText(img, "No sign detected", (x - 120, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)

def draw_history_panel(img, panel_y):
    """Draw history of detected signs"""
    if detected_history:
        # History label
        cv2.putText(img, "History:", (20, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
        
        # Draw history items
        x_offset = 140
        for i, sign in enumerate(detected_history[-MAX_HISTORY:]):
            # Box for each history item
            box_x = x_offset + (i * 70)
            box_y = panel_y + 10
            
            cv2.rectangle(img, (box_x, box_y), (box_x + 60, box_y + 60), 
                         COLOR_SECONDARY, 2)
            
            # Sign letter
            cv2.putText(img, sign, (box_x + 15, box_y + 45), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOR_TEXT, 2)

def draw_stats(img, panel_y, fps, hand_detected):
    """Draw statistics panel"""
    stats_x = img.shape[1] - 300
    stats_y = panel_y + 30
    
    # FPS
    fps_color = COLOR_PRIMARY if fps > 20 else COLOR_ACCENT
    cv2.putText(img, f"FPS: {int(fps)}", (stats_x, stats_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    
    # Hand detection status
    status = "Hand Detected" if hand_detected else "No Hand"
    status_color = COLOR_PRIMARY if hand_detected else COLOR_ACCENT
    cv2.putText(img, f"Status: {status}", (stats_x, stats_y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Instructions
    cv2.putText(img, "Press 'Q' to quit | 'C' to clear", (20, panel_y + 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def draw_typed_text(img, panel_y, text):
    """Show the assembled text"""
    display_text = text[-45:] if len(text) > 45 else text
    x1 = (img.shape[1] // 2) - 250
    y1 = panel_y + 80
    overlay = img.copy()
    draw_rounded_rectangle(overlay, (x1, y1), (x1 + 500, y1 + 70), COLOR_PANEL, -1, 20)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    draw_rounded_rectangle(img, (x1, y1), (x1 + 500, y1 + 70), COLOR_SECONDARY, 2, 20)
    cv2.putText(img, "Typed Text", (x1 + 20, y1 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_TEXT, 2)
    cv2.putText(img, display_text if display_text else "(empty)", (x1 + 20, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

def draw_instruction_panel(img):
    """Display step-by-step instructions"""
    width = 420
    height = 175
    x1, y1 = 20, 90
    x2, y2 = x1 + width, y1 + height
    overlay = img.copy()
    draw_rounded_rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1, 20)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    draw_rounded_rectangle(img, (x1, y1), (x2, y2), COLOR_PRIMARY, 2, 20)
    cv2.putText(img, "Instructions", (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 2)
    for idx, line in enumerate(INSTRUCTION_LINES):
        cv2.putText(img, line, (x1 + 20, y1 + 60 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

def draw_legend_panel(img):
    """Display quick reference legend"""
    width = 300
    height = 160
    x2 = img.shape[1] - 20
    y1 = 90
    x1 = x2 - width
    y2 = y1 + height
    overlay = img.copy()
    draw_rounded_rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL, -1, 20)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    draw_rounded_rectangle(img, (x1, y1), (x2, y2), COLOR_SECONDARY, 2, 20)
    cv2.putText(img, "Legend (Supported Signs)", (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_TEXT, 2)
    for idx, row in enumerate(LEGEND_ROWS):
        cv2.putText(img, row, (x1 + 20, y1 + 60 + idx * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

def draw_hand_indicator(img, posList):
    """Draw a visual indicator when hand is detected"""
    if len(posList) > 0:
        # Get center of hand
        x_coords = [pos[1] for pos in posList]
        y_coords = [pos[2] for pos in posList]
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        
        # Draw pulsing circle
        pulse = int(20 + 10 * abs(np.sin(time.time() * 3)))
        cv2.circle(img, (center_x, center_y), pulse, COLOR_PRIMARY, 2)
        cv2.circle(img, (center_x, center_y), 5, COLOR_PRIMARY, -1)

def detect_sign(posList):
    """Sign detection logic (same as original)"""
    if len(posList) == 0:
        return ""
    
    result = ""
    fingers = []
    
    finger_mcp = [5,9,13,17]
    finger_dip = [6,10,14,18]
    finger_pip = [7,11,15,19]
    finger_tip = [8,12,16,20]
    
    for id in range(4):
        if(posList[finger_tip[id]][1]+ 25  < posList[finger_dip[id]][1] and posList[16][2]<posList[20][2]):
            fingers.append(0.25)
        elif(posList[finger_tip[id]][2] > posList[finger_dip[id]][2]):
            fingers.append(0)
        elif(posList[finger_tip[id]][2] < posList[finger_pip[id]][2]): 
            fingers.append(1)
        elif(posList[finger_tip[id]][1] > posList[finger_pip[id]][1] and posList[finger_tip[id]][1] > posList[finger_dip[id]][1]): 
            fingers.append(0.5)
    
    # Detection conditions (same as original)
    if(posList[3][2] > posList[4][2]) and (posList[3][1] > posList[6][1])and (posList[4][2] < posList[6][2]) and fingers.count(0) == 4:
        result = "A"
    elif(posList[3][1] > posList[4][1]) and fingers.count(1) == 4:
        result = "B"
    elif(posList[3][1] > posList[6][1]) and fingers.count(0.5) >= 1 and (posList[4][2]> posList[8][2]):
        result = "C"
    elif len(fingers) > 0 and (fingers[0]==1) and fingers.count(0) == 3 and (posList[3][1] > posList[4][1]):
        result = "D"
    elif (posList[3][1] < posList[6][1]) and fingers.count(0) == 4 and posList[12][2]<posList[4][2]:
        result = "E"
    elif (fingers.count(1) == 3) and len(fingers) > 0 and (fingers[0]==0) and (posList[3][2] > posList[4][2]):
        result = "F"
    elif len(fingers) > 0 and (fingers[0]==0.25) and fingers.count(0) == 3:
        result = "G"
    elif len(fingers) > 1 and (fingers[0]==0.25) and(fingers[1]==0.25) and fingers.count(0) == 2:
        result = "H"
    elif (posList[4][1] < posList[6][1]) and fingers.count(0) == 3:
        if (len(fingers)==4 and fingers[3] == 1):
            result = "I"
    elif (posList[4][1] < posList[6][1] and posList[4][1] > posList[10][1] and fingers.count(1) == 2):
        result = "K"
    elif len(fingers) > 0 and (fingers[0]==1) and fingers.count(0) == 3 and (posList[3][1] < posList[4][1]):
        result = "L"
    elif (posList[4][1] < posList[16][1]) and fingers.count(0) == 4:
        result = "M"
    elif (posList[4][1] < posList[12][1]) and fingers.count(0) == 4:
        result = "N"
    elif (posList[4][1] > posList[12][1]) and posList[4][2]<posList[6][2] and fingers.count(0) == 4:
        result = "T"
    elif (posList[4][1] > posList[12][1]) and posList[4][2]<posList[12][2] and fingers.count(0) == 4:
        result = "S"
    elif(posList[4][2] < posList[8][2]) and (posList[4][2] < posList[12][2]) and (posList[4][2] < posList[16][2]) and (posList[4][2] < posList[20][2]):
        result = "O"
    elif len(fingers) > 3 and (fingers[2] == 0) and (posList[4][2] < posList[12][2]) and (posList[4][2] > posList[6][2]):
        if (len(fingers)==4 and fingers[3] == 0):
            result = "P"
    elif len(fingers) > 3 and (fingers[1] == 0) and (fingers[2] == 0) and (fingers[3] == 0) and (posList[8][2] > posList[5][2]) and (posList[4][2] < posList[1][2]):
        result = "Q"
    elif(posList[8][1] < posList[12][1]) and (fingers.count(1) == 2) and (posList[9][1] > posList[4][1]):
        result = "R"
    elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2] and (posList[8][1] - posList[11][1]) <= 50):
        result = "U"
    elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2]):
        result = "V"
    elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 3):
        result = "W"
    elif len(fingers) > 0 and (fingers[0] == 0.5) and fingers.count(0) == 3 and posList[4][1] > posList[6][1]:
        result = "X"
    elif(fingers.count(0) == 3) and (posList[3][1] < posList[4][1]):
        if (len(fingers)==4 and fingers[3] == 1):
            result = "Y"
    
    return result

# Main loop
pTime = 0
last_result = ""
result_stable_count = 0
STABLE_FRAMES = 10  # Number of frames to confirm detection

print("Enhanced Sign Language Detection Started")
print("Press 'Q' to quit, 'C' to clear history")

while True:
    try:
        success, img = cap.read()
        if not success:
            continue
        
        # Flip image for mirror effect
        img = cv2.flip(img, 1)
        
        # Detect hands
        img = detector.findHands(img, draw=True)
        posList = detector.findPosition(img, draw=False)
        
        # Detect sign
        current_result = detect_sign(posList)
        
        # Stabilize detection (only add if detected for multiple frames)
        if current_result:
            if current_result == last_result:
                result_stable_count += 1
                if result_stable_count >= STABLE_FRAMES:
                    if not detected_history or detected_history[-1] != current_result:
                        detected_history.append(current_result)
                        typed_text += current_result
                        if len(detected_history) > MAX_HISTORY:
                            detected_history.pop(0)
                    result_stable_count = 0
            else:
                result_stable_count = 0
                last_result = current_result
        else:
            result_stable_count = 0
            last_result = ""
        
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        
        # Draw UI elements
        draw_header(img)
        draw_hand_indicator(img, posList)
        panel_y = draw_ui_panel(img)
        draw_detected_sign(img, current_result, img.shape[1] // 2, 150)
        draw_history_panel(img, panel_y)
        draw_stats(img, panel_y, avg_fps, len(posList) > 0)
        draw_typed_text(img, panel_y, typed_text)
        draw_instruction_panel(img)
        draw_legend_panel(img)
        
        # Display
        cv2.imshow("ASL Sign Language Detection - Enhanced UI", img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            detected_history.clear()
            typed_text = ""
            print("History cleared")
        elif key == ord(' '):
            typed_text += ' '
            detected_history.append('[Space]')
            if len(detected_history) > MAX_HISTORY:
                detected_history.pop(0)
            
    except (IndexError, KeyError) as e:
        print(f"Detection error (continuing): {e}")
        continue
    except KeyboardInterrupt:
        print("\nExiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Application closed successfully")
