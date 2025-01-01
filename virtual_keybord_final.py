import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
keyboard = Controller()

# Set up the Mediapipe Hands object
def detect_hand_movement():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Define keyboard layout
    keyboard_layout = {
        (50 + j * 100, 100 + i * 100): key
        for i, row in enumerate([
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", "BACK"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "RESET"]
        ])
        for j, key in enumerate(row)
    }

    selected_text = ""
    last_pressed_time = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty frame.")
                continue

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Draw keyboard layout
            for pos, char in keyboard_layout.items():
                x, y = pos
                cv2.rectangle(frame, (x - 40, y - 40), (x + 40, y + 40), (255, 255, 255), -1)
                cv2.putText(frame, char, (x - 20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Convert BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get landmark of index finger tip
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cursor_x = int(index_finger_tip.x * w)
                    cursor_y = int(index_finger_tip.y * h)

                    # Visualize cursor
                    cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 0, 0), -1)

                    # Check if cursor is on a key
                    for pos, char in keyboard_layout.items():
                        x, y = pos
                        if x - 40 < cursor_x < x + 40 and y - 40 < cursor_y < y + 40:
                            # Debounce keypress
                            if time.time() - last_pressed_time > 0.15:
                                if char == "BACKSPACE":
                                    selected_text = selected_text[:-1]
                                elif char == "RESET":
                                    selected_text = ""
                                else:
                                    selected_text += char
                                    keyboard.press(char)

                                last_pressed_time = time.time()
                            cv2.rectangle(frame, (x - 40, y - 40), (x + 40, y + 40), (0, 255, 0), 3)

            # Display selected text
            cv2.putText(frame, f"Typed: {selected_text}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the output
            cv2.imshow('Virtual Keyboard', frame)

            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hand_movement()
