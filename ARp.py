import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe face detection and hands module
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the glasses images with alpha channels (RGBA) with specified values
glasses_images = {
    'glasses.png': {'path': 'glasses.png', 'x_adjust': -400, 'y_adjust': -60},
    'circle_glasses.png': {'path': 'circle_glasses.png', 'x_adjust': -450, 'y_adjust': -200},
    'flower_glasses.png': {'path': 'flower_glasses.png', 'x_adjust': -450, 'y_adjust': -150}
}
glasses_index = 0
current_glasses_key = list(glasses_images.keys())[glasses_index]
glasses = cv2.imread(glasses_images[current_glasses_key]['path'], cv2.IMREAD_UNCHANGED)

# Load the QR code image
qr_code = cv2.imread('qr_code.png', cv2.IMREAD_UNCHANGED)

def overlay_image(frame, overlay_img, x, y, scale=1.0):
    overlay_width = int(overlay_img.shape[1] * scale)
    overlay_height = int(overlay_img.shape[0] * scale)
    overlay_resized = cv2.resize(overlay_img, (overlay_width, overlay_height))

    y1, y2 = y, y + overlay_resized.shape[0]
    x1, x2 = x, x + overlay_resized.shape[1]

    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        return frame

    alpha_s = overlay_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * overlay_resized[:, :, c] +
                                  alpha_l * frame[y1:y2, x1:x2, c])

    return frame

def overlay_glasses(frame, glasses, x, y, w, h, scale=2.0, x_adjust=0, y_adjust=0):
    return overlay_image(frame, glasses, x + x_adjust, y + y_adjust, scale)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection, \
    mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    hand_open = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_detection.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                y -= int(h / 4)
                frame = overlay_glasses(frame, glasses, x, y, w, h, scale=2,
                                        x_adjust=glasses_images[current_glasses_key]['x_adjust'],
                                        y_adjust=glasses_images[current_glasses_key]['y_adjust'])

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                thumb_is_open = thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
                fingers_are_open = all([
                    index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                    middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                    ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                    pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
                ])

                if thumb_is_open and fingers_are_open:
                    if not hand_open:
                        glasses_index = (glasses_index + 1) % len(glasses_images)
                        current_glasses_key = list(glasses_images.keys())[glasses_index]
                        glasses = cv2.imread(glasses_images[current_glasses_key]['path'], cv2.IMREAD_UNCHANGED)
                        hand_open = True
                else:
                    hand_open = False

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Overlay QR code on the side of the frame
        frame_height, frame_width, _ = frame.shape
        qr_code_x = frame_width - qr_code.shape[1] - 10  # 10 pixels from the right edge
        qr_code_y = 10  # 10 pixels from the top edge
        frame = overlay_image(frame, qr_code, qr_code_x, qr_code_y, scale=0.5)  # Adjust scale as needed

        cv2.imshow('AR Glasses', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()