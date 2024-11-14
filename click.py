import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

sensitivity = 1.5 
click_threshold = 20  

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
               
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                x_index = int(index_finger_tip.x * frame_width)
                y_index = int(index_finger_tip.y * frame_height)

                x_thumb = int(thumb_tip.x * frame_width)
                y_thumb = int(thumb_tip.y * frame_height)

                distance = math.hypot(x_thumb - x_index, y_thumb - y_index)

                
                screen_x = int(index_finger_tip.x * screen_width * sensitivity)
                screen_y = int(index_finger_tip.y * screen_height * sensitivity)

                
                pyautogui.moveTo(screen_x, screen_y)

                
                if distance < click_threshold:
                    
                    pyautogui.click()
                
                    pyautogui.sleep(0.2)

                cv2.circle(frame, (x_index, y_index), 10, (255, 0, 255), -1)
                cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 255, 255), -1)

        cv2.imshow("Hand Tracking - Mouse Control with Click", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
