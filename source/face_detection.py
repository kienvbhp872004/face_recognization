import random

import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import os


# Hàm tính khoảng cách Euclidean giữa hai điểm
def euclidean(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def check_blurriness_gaussian(image, threshold=10):

    if image is None:
        print("Không thể đọc ảnh!")
        return None

    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    diff = cv2.absdiff(image, blurred)
    mean_diff = np.mean(diff)

    print(f"Độ sắc nét (Gaussian Diff): {mean_diff:.2f}")

    # if mean_diff < threshold:
    #     return "Ảnh bị mờ!"
    # else:
    #     return "Ảnh sắc nét!"
    return mean_diff


import cv2
import numpy as np


def check_blurriness_tenengrad(image, threshold=500):
    if image is None:
        print("Không thể đọc ảnh!")
        return None

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    sharpness = np.mean(gradient_magnitude)  # Tính trung bình độ sắc nét

    print(f"Độ sắc nét (Tenengrad): {sharpness:.2f}")

    if sharpness < threshold:
        print("Ảnh bị mờ!")
    else:
        print("Ảnh sắc nét!")

    return sharpness


# Kiểm tra ảnh





# Kiểm tra ảnh
image_path = "path/to/image.jpg"

# Tính Eye Aspect Ratio (EAR) để phát hiện chớp mắt
def EAR(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# Cấu hình tham số
CAM_WIDTH, CAM_HEIGHT = 640, 480
THRESHOLD_EAR = 0.2  # Ngưỡng phát hiện chớp mắt
PITCH_THRESHOLD = 10  # Ngưỡng phát hiện ngửa mặt
YAW_THRESHOLD = 20
BALANCE_THRESHOLD = 35
offsetX = 0.2
offsetY = 0.3
offsetYaw = 45
FRAME_THRESHOLD = 3
closedEyes = False
counterBlink = 0
authentication = True
user_id = "20224869"
save_dir = os.path.join("data\image", user_id)
os.makedirs(save_dir, exist_ok=True)
# Khởi tạo webcam và detector
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
detector = FaceMeshDetector(maxFaces=1)

# Chỉ số các điểm đặc trưng trên mặt
IDXS = {
    "nose": 1, "chin": 199,
    "left_eye": 33, "right_eye": 263,
    "left_mouth": 61, "right_mouth": 291
}


# Hàm tính toán
def state_face(pitch, yaw):
    if pitch > PITCH_THRESHOLD:
        return "Head turned right"
    elif pitch < -PITCH_THRESHOLD:
        return "Head turned left"
    elif yaw > YAW_THRESHOLD:
        return "Head tilted up"
    elif abs(yaw) < BALANCE_THRESHOLD and abs(pitch) > -BALANCE_THRESHOLD:
        return "Head facing forward"


while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        x_vals = [p[0] for p in face]
        y_vals = [p[1] for p in face]

        # Tính bbox
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_min_scale = int(x_min - offsetX * (x_max - x_min))
        y_min_scale = int(y_min - offsetY * (y_max - y_min))
        x_max_scale = int(x_max + offsetX * (x_max - x_min))
        y_max_scale = int(y_max + 0.5 * offsetY * (y_max - y_min))
        # Vẽ bounding box lên ảnh
        cv2.rectangle(img, (x_min_scale, y_min_scale), (x_max_scale, y_max_scale), (0, 255, 0), 2)
        # Lấy các điểm mắt
        leftEye = [face[i] for i in [33, 160, 158, 133, 153, 144]]
        rightEye = [face[i] for i in [362, 385, 387, 263, 373, 380]]

        # Tính EAR cho từng mắt
        ear_left = EAR(leftEye)
        ear_right = EAR(rightEye)

        # Lấy các điểm mốc
        points = {key: face[value] for key, value in IDXS.items()}

        # Tính pitch (ngửa/cúi đầu) và yaw (quay trái/phải)
        nose_chin = np.array(points["chin"]) - np.array(points["nose"])
        nose_left_eye = np.array(points["nose"]) - np.array(points["left_eye"])

        pitch = np.arctan2(nose_chin[0], nose_chin[1]) * (180 / np.pi)
        yaw = np.arctan2(nose_left_eye[0], nose_left_eye[1]) * (180 / np.pi) - offsetYaw

        # Kiểm tra chớp mắt
        if ear_left < THRESHOLD_EAR and ear_right < THRESHOLD_EAR and not closedEyes:
            closedEyes = True
        if closedEyes and ear_left > THRESHOLD_EAR and ear_right > THRESHOLD_EAR:
            print("Chớp mắt")
            closedEyes = False
        for key, (x, y) in points.items():
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(img, state_face(pitch, yaw), (x_min_scale , y_min_scale - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Face Mesh", img)

        ## Chuẩn hóa khuôn mặt
        w = x_max_scale - x_min_scale
        h = y_max_scale - y_min_scale
        x_top_left = x_min_scale / CAM_WIDTH
        y_top_left = y_min_scale / CAM_HEIGHT
        x_bottom_right = x_min_scale / CAM_WIDTH
        y_bottom_right = y_min_scale / CAM_HEIGHT
        ## Lưu ảnh
        if authentication:
            face = img[y_min_scale:y_max_scale, x_min_scale:x_max_scale]
            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_path = os.path.join(save_dir, f"{user_id}_{state_face(pitch, yaw)}_{check_blurriness_tenengrad(face)}_{random.randint(0,100)}.jpg")
                cv2.imwrite(img_path, face)
                print(f"Image saved to {img_path}")
                # authentication = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
