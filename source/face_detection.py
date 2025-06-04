import os
import cv2
import numpy as np
import random

from PIL import Image
from tensorflow.keras.models import load_model, Model
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from cvzone.FaceMeshModule import FaceMeshDetector
import joblib

### ========== CẤU HÌNH ==========
CAM_WIDTH, CAM_HEIGHT = 640, 480
THRESHOLD_EAR = 0.2
PITCH_THRESHOLD = 10
YAW_THRESHOLD = 20
BALANCE_THRESHOLD = 35
THRESHOLD_AUTHENTICATION = 40
BRIGHTNESS_THRESHOLD = 100
BLUR_SOBEL_THRESHOLD = 20
FRAME_THRESHOLD = 3
offsetX, offsetY, offsetYaw = 0.2, 0.3, 45

user_id = "2"
save_dir = os.path.join("data_raw", "image", user_id)
os.makedirs(save_dir, exist_ok=True)
model_type = "facenet"  # Chọn mô hình: "resnet50" hoặc "facenet"

### ========== LOAD MODEL ==========
# Load resnet50 (TensorFlow)
try:
    model = load_model(
        r'C:\Users\admin\OneDrive - Hanoi University of Science and Technology\Documents\GitHub\PTTK\face_recognization\model\resnet50.h5')
    embedding_model = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
except Exception as e:
    print("Lỗi khi load model resnet50:", e)
    exit()

# Load facenet (PyTorch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').to(device)
resnet.eval()

# Preprocessing pipeline cho facenet
preprocess_facenet = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_embeddings_and_model(embed_path='embeddings.npy',
                              label_path='labels.npy',
                              model_path='svm_model.joblib',
                              encoder_path='label_encoder.joblib'):
    """Tải embedding, nhãn, mô hình SVM và LabelEncoder."""
    if not all(os.path.exists(path) for path in [embed_path, label_path, model_path, encoder_path]):
        raise FileNotFoundError(
            f"One or more files not found: {embed_path}, {label_path}, {model_path}, {encoder_path}")

    X_embed = np.load(embed_path)
    y = np.load(label_path)
    svm_model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    print(f"Loaded embeddings from {embed_path}, shape: {X_embed.shape}")
    print(f"Loaded labels from {label_path}, shape: {y.shape}")
    print(f"Loaded SVM model from {model_path}")
    print(f"Loaded LabelEncoder from {encoder_path}")

    return X_embed, y, svm_model, label_encoder


# Tải embedding và nhãn cho cả hai mô hình
saved_embeddings = np.load('user_embeddings.npy')
saved_labels = np.load('user_labels.npy', allow_pickle=True)
if model_type == 'facenet':
    X_embed_loaded, y_loaded, svm_model_loaded, label_encoder_loaded = load_embeddings_and_model(
        embed_path='embeddings.npy',
        label_path='labels.npy',
        model_path='svm_model.joblib',
        encoder_path='label_encoder.joblib'
    )


### ========== HÀM DỰ ĐOÁN ==========
def predict_face(image_input, threshold=0.4):
    """
    Dự đoán nhãn và độ tự tin từ ảnh khuôn mặt.

    Parameters:
    - image_input: Mảng NumPy (H, W, 3)
    - threshold: Ngưỡng độ tự tin cho resnet50 (cosine similarity)

    Returns:
    - label: Nhãn dự đoán (string, ví dụ: 'person1' hoặc 'Unknown')
    - confidence: Độ tự tin (float, trong khoảng [0, 1])
    """
    if not isinstance(image_input, np.ndarray) or image_input.ndim != 3 or image_input.shape[-1] != 3:
        raise ValueError(
            f"Invalid image_input: Expected NumPy array with shape (H, W, 3), got {image_input.shape if isinstance(image_input, np.ndarray) else type(image_input)}")

    if model_type == "resnet50":
        # Preprocessing cho resnet50
        img = cv2.resize(image_input, (112, 112))
        img = img.astype(np.float32) / 255.0
        emb = embedding_model.predict(img[np.newaxis])[0]
        similarities = cosine_similarity([emb], saved_embeddings)[0]
        best_idx = np.argmax(similarities)
        confidence = similarities[best_idx]
        label = saved_labels[best_idx] if confidence >= threshold else "Unknown"
    elif model_type == "facenet":
        # Preprocessing cho facenet
        if image_input.shape[0] in [1, 3]:  # (C, H, W)
            image_input = image_input.transpose(1, 2, 0)  # Chuyển thành (H, W, C)
        if image_input.dtype != np.uint8:
            image_input = (image_input * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_input)
        image_tensor = preprocess_facenet(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(image_tensor).cpu().detach().numpy().squeeze(0)
        pred = svm_model_loaded.predict([embedding])
        pred_proba = svm_model_loaded.predict_proba([embedding])
        label = label_encoder_loaded.inverse_transform(pred)[0]
        confidence = np.max(pred_proba, axis=1)[0]
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Expected 'resnet50' or 'facenet'")

    return label, confidence


### ========== HÀM HỖ TRỢ ==========
def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def EAR(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def state_face(pitch, yaw):
    if yaw > YAW_THRESHOLD:
        return "Head tilted up"
    elif pitch > PITCH_THRESHOLD:
        return "Head turned right"
    elif pitch < -PITCH_THRESHOLD:
        return "Head turned left"
    elif abs(yaw) < BALANCE_THRESHOLD:
        return "Head facing forward"
    return "Unknown"


def detect_blur_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    return np.mean(gradient_magnitude)


def brightness_mean(image):
    return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


### ========== KHỞI TẠO ==========
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
detector = FaceMeshDetector(maxFaces=1)
IDXS = {"nose": 1, "chin": 199, "left_eye": 33, "right_eye": 263, "left_mouth": 61, "right_mouth": 291}
closedEyes = False
authentication = False

### ========== VÒNG LẶP CHÍNH ==========
while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        x_vals = [p[0] for p in face]
        y_vals = [p[1] for p in face]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_min_s, y_min_s = int(x_min - offsetX * (x_max - x_min)), int(y_min - offsetY * (y_max - y_min))
        x_max_s, y_max_s = int(x_max + offsetX * (x_max - x_min)), int(y_max + 0.5 * offsetY * (y_max - y_min))

        face_crop = img[y_min_s:y_max_s, x_min_s:x_max_s]
        if face_crop.size == 0:
            continue

        leftEye = [face[i] for i in [33, 160, 158, 133, 153, 144]]
        rightEye = [face[i] for i in [362, 385, 387, 263, 373, 380]]
        ear_left, ear_right = EAR(leftEye), EAR(rightEye)

        points = {key: face[value] for key, value in IDXS.items()}
        pitch = np.arctan2(points["chin"][0] - points["nose"][0], points["chin"][1] - points["nose"][1]) * (180 / np.pi)
        yaw = np.arctan2(points["nose"][0] - points["left_eye"][0], points["nose"][1] - points["left_eye"][1]) * (
                    180 / np.pi) - offsetYaw

        # Blink detection (giữ nguyên)
        # if ear_left < THRESHOLD_EAR and ear_right < THRESHOLD_EAR and not closedEyes:
        #     closedEyes = True
        # elif closedEyes and ear_left > THRESHOLD_EAR and ear_right > THRESHOLD_EAR:
        #     print("Chớp mắt")
        #     closedEyes = False

        cv2.rectangle(img, (x_min_s, y_min_s), (x_max_s, y_max_s), (0, 255, 0), 2)
        state = state_face(pitch, yaw)

        if authentication:
            if cv2.waitKey(1) & 0xFF == ord('s'):
                brightness = brightness_mean(face_crop)
                blur = detect_blur_sobel(face_crop)
                if brightness > BRIGHTNESS_THRESHOLD and blur > BLUR_SOBEL_THRESHOLD:
                    img_path = os.path.join(save_dir, f"{user_id}_{state}_{blur:.2f}_{random.randint(0, 100)}.jpg")
                    cv2.imwrite(img_path, face_crop)
                    print(f"Image saved to {img_path}")
            cv2.putText(img, f"{state} | Bright: {brightness_mean(face_crop):.1f}", (x_min_s, y_min_s - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            label, conf = predict_face(face_crop)
            cv2.putText(img, f"User: {label} | Confidence: {conf:.2f}", (x_min_s, y_min_s - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Face Mesh", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()