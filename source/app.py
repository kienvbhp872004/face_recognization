import os
import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from PIL import Image, ImageTk

from tensorflow.keras.models import load_model, Model
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from cvzone.FaceMeshModule import FaceMeshDetector
import joblib


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        # Thiết lập biến
        self.CAM_WIDTH, self.CAM_HEIGHT = 640, 480
        self.THRESHOLD_EAR = 0.2
        self.PITCH_THRESHOLD = 10
        self.YAW_THRESHOLD = 20
        self.BALANCE_THRESHOLD = 35
        self.THRESHOLD_AUTHENTICATION = 40
        self.BRIGHTNESS_THRESHOLD = 100
        self.BLUR_SOBEL_THRESHOLD = 20
        self.FRAME_THRESHOLD = 3
        self.offsetX, self.offsetY, self.offsetYaw = 0.2, 0.3, 45

        self.user_id = "2"
        self.save_dir = os.path.join("data_raw/data_raw", "image", self.user_id)
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_type = "facenet"  # Chọn mô hình: "resnet50" hoặc "facenet"

        # Trạng thái
        self.is_running = False
        self.authentication = False
        self.closedEyes = False
        self.cap = None
        self.detector = None
        self.current_frame = None
        self.IDXS = {"nose": 1, "chin": 199, "left_eye": 33, "right_eye": 263, "left_mouth": 61, "right_mouth": 291}

        # Tạo UI
        self.create_ui()

        # Load models
        self.load_models()

    def create_ui(self):
        # Tạo frame chính
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Frame bên trái - video stream
        self.left_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Canvas cho video
        self.canvas = tk.Canvas(self.left_frame, width=self.CAM_WIDTH, height=self.CAM_HEIGHT, bg="black")
        self.canvas.pack(padx=10, pady=10)

        # Frame bên phải - điều khiển
        right_frame = ttk.LabelFrame(main_frame, text="Controls")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Frame cho thông tin người dùng
        user_frame = ttk.LabelFrame(right_frame, text="User Information")
        user_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(user_frame, text="User ID:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.user_id_entry = ttk.Entry(user_frame, width=20)
        self.user_id_entry.grid(row=0, column=1, padx=5, pady=5)
        self.user_id_entry.insert(0, self.user_id)

        ttk.Label(user_frame, text="Model Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_type_combo = ttk.Combobox(user_frame, values=["resnet50", "facenet"], state="readonly", width=17)
        self.model_type_combo.grid(row=1, column=1, padx=5, pady=5)
        self.model_type_combo.set(self.model_type)

        # Frame cho thông số
        param_frame = ttk.LabelFrame(right_frame, text="Parameters")
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        # Brightness threshold
        ttk.Label(param_frame, text="Brightness Threshold:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.brightness_scale = ttk.Scale(param_frame, from_=50, to=200, orient=tk.HORIZONTAL, length=150)
        self.brightness_scale.grid(row=0, column=1, padx=5, pady=5)
        self.brightness_scale.set(self.BRIGHTNESS_THRESHOLD)
        self.brightness_value = ttk.Label(param_frame, text=str(self.BRIGHTNESS_THRESHOLD))
        self.brightness_value.grid(row=0, column=2, padx=5, pady=5)

        # Blur threshold
        ttk.Label(param_frame, text="Blur Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.blur_scale = ttk.Scale(param_frame, from_=5, to=50, orient=tk.HORIZONTAL, length=150)
        self.blur_scale.grid(row=1, column=1, padx=5, pady=5)
        self.blur_scale.set(self.BLUR_SOBEL_THRESHOLD)
        self.blur_value = ttk.Label(param_frame, text=str(self.BLUR_SOBEL_THRESHOLD))
        self.blur_value.grid(row=1, column=2, padx=5, pady=5)

        # Confidence threshold
        ttk.Label(param_frame, text="Confidence Threshold:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.conf_scale = ttk.Scale(param_frame, from_=0.3, to=0.9, orient=tk.HORIZONTAL, length=150)
        self.conf_scale.grid(row=2, column=1, padx=5, pady=5)
        self.conf_scale.set(0.4)
        self.conf_value = ttk.Label(param_frame, text="0.40")
        self.conf_value.grid(row=2, column=2, padx=5, pady=5)

        # Cập nhật giá trị khi scale thay đổi
        self.brightness_scale.config(command=lambda s: self.update_scale_value(s, self.brightness_value, "brightness"))
        self.blur_scale.config(command=lambda s: self.update_scale_value(s, self.blur_value, "blur"))
        self.conf_scale.config(command=lambda s: self.update_scale_value(s, self.conf_value, "conf"))

        # Frame cho nút điều khiển
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.mode_button = ttk.Button(button_frame, text="Recognition Mode", command=self.toggle_mode)
        self.mode_button.grid(row=0, column=1, padx=5, pady=5)

        self.capture_button = ttk.Button(button_frame, text="Capture Image", command=self.capture_image,
                                         state=tk.DISABLED)
        self.capture_button.grid(row=0, column=2, padx=5, pady=5)

        # Frame trạng thái
        status_frame = ttk.LabelFrame(right_frame, text="Status")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status_text = tk.Text(status_frame, height=10, width=40, wrap=tk.WORD)
        self.status_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.status_text.config(state=tk.DISABLED)

        # Cân chỉnh grid
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

    def update_scale_value(self, value, label_widget, type_scale):
        if type_scale == "brightness":
            self.BRIGHTNESS_THRESHOLD = int(float(value))
            label_widget.config(text=str(self.BRIGHTNESS_THRESHOLD))
        elif type_scale == "blur":
            self.BLUR_SOBEL_THRESHOLD = int(float(value))
            label_widget.config(text=str(self.BLUR_SOBEL_THRESHOLD))
        elif type_scale == "conf":
            label_widget.config(text=f"{float(value):.2f}")

    def log_status(self, message):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def load_models(self):
        try:
            self.log_status("Loading models...")

            # Load resnet50 (TensorFlow)
            try:
                self.model = load_model(r'C:\Users\admin\OneDrive - Hanoi University of Science and Technology\Documents\GitHub\PTTK\face_recognization\model\my_model.h5')
                self.embedding_model = Model(inputs=self.model.input, outputs=self.model.get_layer('flatten_1').output)
                self.log_status("ResNet50 model loaded successfully")
            except Exception as e:
                self.log_status(f"Error loading ResNet50 model: {e}")

            # Load facenet (PyTorch)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_status(f"Using device: {self.device}")
            self.resnet = InceptionResnetV1(pretrained='vggface2').to(self.device)
            self.resnet.eval()

            # Preprocessing pipeline cho facenet
            self.preprocess_facenet = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            # Tải embedding và nhãn
            self.saved_embeddings = np.load('user_embeddings.npy')
            self.saved_labels = np.load('user_labels.npy', allow_pickle=True)

            if self.model_type == 'facenet':
                self.X_embed_loaded, self.y_loaded, self.svm_model_loaded, self.label_encoder_loaded = self.load_embeddings_and_model(
                    embed_path='embeddings.npy',
                    label_path='labels.npy',
                    model_path='svm_model.joblib',
                    encoder_path='label_encoder.joblib'
                )

            self.log_status("All models loaded successfully!")

        except Exception as e:
            self.log_status(f"Error during model loading: {e}")
            messagebox.showerror("Model Loading Error", f"Failed to load models: {e}")

    def load_embeddings_and_model(self, embed_path='embeddings.npy', label_path='labels.npy',
                                  model_path='svm_model.joblib', encoder_path='label_encoder.joblib'):
        """Tải embedding, nhãn, mô hình SVM và LabelEncoder."""
        if not all(os.path.exists(path) for path in [embed_path, label_path, model_path, encoder_path]):
            error_msg = f"One or more files not found: {embed_path}, {label_path}, {model_path}, {encoder_path}"
            self.log_status(error_msg)
            raise FileNotFoundError(error_msg)

        X_embed = np.load(embed_path)
        y = np.load(label_path)
        svm_model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)

        self.log_status(f"Loaded embeddings from {embed_path}, shape: {X_embed.shape}")
        self.log_status(f"Loaded labels from {label_path}, shape: {y.shape}")
        self.log_status(f"Loaded SVM model from {model_path}")
        self.log_status(f"Loaded LabelEncoder from {encoder_path}")

        return X_embed, y, svm_model, label_encoder

    def toggle_camera(self):
        if not self.is_running:
            # Cập nhật user_id
            self.user_id = self.user_id_entry.get()
            self.save_dir = os.path.join("data_raw/data_raw", "image", self.user_id)
            os.makedirs(self.save_dir, exist_ok=True)

            # Cập nhật model_type
            self.model_type = self.model_type_combo.get()

            # Bắt đầu camera
            self.is_running = True
            self.start_button.config(text="Stop Camera")
            self.capture_button.config(state=tk.NORMAL)

            # Khởi tạo camera
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, self.CAM_WIDTH)
            self.cap.set(4, self.CAM_HEIGHT)
            self.detector = FaceMeshDetector(maxFaces=1)

            # Bắt đầu thread xử lý video
            self.video_thread = threading.Thread(target=self.process_video)
            self.video_thread.daemon = True
            self.video_thread.start()

            self.log_status(f"Camera started - User ID: {self.user_id}, Model: {self.model_type}")
        else:
            # Dừng camera
            self.is_running = False
            self.start_button.config(text="Start Camera")
            self.capture_button.config(state=tk.DISABLED)

            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.log_status("Camera stopped")

    def toggle_mode(self):
        self.authentication = not self.authentication
        if self.authentication:
            self.mode_button.config(text="Registration Mode")
            self.log_status("Switched to Registration Mode")
        else:
            self.mode_button.config(text="Recognition Mode")
            self.log_status("Switched to Recognition Mode")

    def process_video(self):
        while self.is_running and self.cap is not None:
            success, img = self.cap.read()
            if not success:
                self.log_status("Failed to read from camera")
                break

            img, faces = self.detector.findFaceMesh(img, draw=False)
            face_crop = None

            if faces:
                face = faces[0]
                x_vals = [p[0] for p in face]
                y_vals = [p[1] for p in face]
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)
                x_min_s, y_min_s = int(x_min - self.offsetX * (x_max - x_min)), int(
                    y_min - self.offsetY * (y_max - y_min))
                x_max_s, y_max_s = int(x_max + self.offsetX * (x_max - x_min)), int(
                    y_max + 0.5 * self.offsetY * (y_max - y_min))

                # Kiểm tra và đảm bảo tọa độ hợp lệ
                x_min_s = max(0, x_min_s)
                y_min_s = max(0, y_min_s)
                x_max_s = min(img.shape[1], x_max_s)
                y_max_s = min(img.shape[0], y_max_s)

                face_crop = img[y_min_s:y_max_s, x_min_s:x_max_s]
                if face_crop.size == 0:
                    face_crop = None
                else:
                    # Phân tích khuôn mặt
                    leftEye = [face[i] for i in [33, 160, 158, 133, 153, 144]]
                    rightEye = [face[i] for i in [362, 385, 387, 263, 373, 380]]
                    ear_left, ear_right = self.EAR(leftEye), self.EAR(rightEye)

                    points = {key: face[value] for key, value in self.IDXS.items()}
                    pitch = np.arctan2(points["chin"][0] - points["nose"][0], points["chin"][1] - points["nose"][1]) * (
                                180 / np.pi)
                    yaw = np.arctan2(points["nose"][0] - points["left_eye"][0],
                                     points["nose"][1] - points["left_eye"][1]) * (180 / np.pi) - self.offsetYaw

                    # Blink detection
                    if ear_left < self.THRESHOLD_EAR and ear_right < self.THRESHOLD_EAR and not self.closedEyes:
                        self.closedEyes = True
                    elif self.closedEyes and ear_left > self.THRESHOLD_EAR and ear_right > self.THRESHOLD_EAR:
                        self.log_status("Blink detected")
                        self.closedEyes = False

                    cv2.rectangle(img, (x_min_s, y_min_s), (x_max_s, y_max_s), (0, 255, 0), 2)
                    state = self.state_face(pitch, yaw)

                    brightness = self.brightness_mean(face_crop)
                    blur = self.detect_blur_sobel(face_crop)

                    if self.authentication:
                        cv2.putText(img, f"{state} | Bright: {brightness:.1f} | Blur: {blur:.1f}",
                                    (x_min_s, y_min_s - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)

                        # Lưu thông tin hiện tại cho nút capture
                        self.current_frame = {
                            'img': face_crop.copy() if face_crop is not None else None,
                            'state': state,
                            'blur': blur,
                            'brightness': brightness
                        }
                    else:
                        if face_crop is not None:
                            threshold = float(self.conf_scale.get())
                            label, conf = self.predict_face(face_crop, threshold)
                            cv2.putText(img, f"User: {label} | Conf: {conf:.2f}",
                                        (x_min_s, y_min_s - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 0, 255), 2)

            # Chuyển đổi OpenCV BGR -> RGB cho hiển thị
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Hiển thị trên canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Giữ tham chiếu

    def capture_image(self):
        if not self.authentication or self.current_frame is None or self.current_frame['img'] is None:
            messagebox.showwarning("Warning",
                                   "Can't capture image. Please ensure you're in Registration Mode and a face is detected.")
            return

        face_crop = self.current_frame['img']
        state = self.current_frame['state']
        blur = self.current_frame['blur']
        brightness = self.current_frame['brightness']

        if brightness > self.BRIGHTNESS_THRESHOLD and blur > self.BLUR_SOBEL_THRESHOLD:
            img_path = os.path.join(self.save_dir, f"{self.user_id}_{state}_{blur:.2f}_{random.randint(0, 100)}.jpg")
            cv2.imwrite(img_path, face_crop)
            self.log_status(f"Image saved to {img_path}")
            messagebox.showinfo("Success", f"Image saved successfully!\nPath: {img_path}")
        else:
            self.log_status(f"Image quality too low - Brightness: {brightness:.1f}, Blur: {blur:.1f}")
            messagebox.showwarning("Image Quality",
                                   f"Image quality is too low.\nBrightness: {brightness:.1f} (min: {self.BRIGHTNESS_THRESHOLD})\nBlur: {blur:.1f} (min: {self.BLUR_SOBEL_THRESHOLD})")

    def predict_face(self, image_input, threshold=0.4):
        """
        Dự đoán nhãn và độ tự tin từ ảnh khuôn mặt.
        """
        if not isinstance(image_input, np.ndarray) or image_input.ndim != 3 or image_input.shape[-1] != 3:
            self.log_status(
                f"Invalid image input shape: {image_input.shape if isinstance(image_input, np.ndarray) else type(image_input)}")
            return "Unknown", 0.0

        try:
            if self.model_type == "resnet50":
                # Preprocessing cho resnet50
                img = cv2.resize(image_input, (112, 112))
                img = img.astype(np.float32) / 255.0
                emb = self.embedding_model.predict(img[np.newaxis])[0]
                similarities = cosine_similarity([emb], self.saved_embeddings)[0]
                best_idx = np.argmax(similarities)
                confidence = similarities[best_idx]
                label = self.saved_labels[best_idx] if confidence >= threshold else "Unknown"
            elif self.model_type == "facenet":
                # Preprocessing cho facenet
                if image_input.shape[0] in [1, 3]:  # (C, H, W)
                    image_input = image_input.transpose(1, 2, 0)  # Chuyển thành (H, W, C)
                if image_input.dtype != np.uint8:
                    image_input = (image_input * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(image_input)
                image_tensor = self.preprocess_facenet(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.resnet(image_tensor).cpu().detach().numpy().squeeze(0)
                pred = self.svm_model_loaded.predict([embedding])
                pred_proba = self.svm_model_loaded.predict_proba([embedding])
                label = self.label_encoder_loaded.inverse_transform(pred)[0]
                confidence = np.max(pred_proba, axis=1)[0]
            else:
                raise ValueError(f"Invalid model_type: {self.model_type}. Expected 'resnet50' or 'facenet'")

            return label, confidence

        except Exception as e:
            self.log_status(f"Error in face prediction: {e}")
            return "Error", 0.0

    # Các hàm hỗ trợ từ mã nguồn gốc
    def euclidean(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def EAR(self, eye):
        A = self.euclidean(eye[1], eye[5])
        B = self.euclidean(eye[2], eye[4])
        C = self.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def state_face(self, pitch, yaw):
        if yaw > self.YAW_THRESHOLD:
            return "Head tilted up"
        elif pitch > self.PITCH_THRESHOLD:
            return "Head turned right"
        elif pitch < -self.PITCH_THRESHOLD:
            return "Head turned left"
        elif abs(yaw) < self.BALANCE_THRESHOLD:
            return "Head facing forward"
        return "Unknown"

    def detect_blur_sobel(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return np.mean(gradient_magnitude)

    def brightness_mean(self, image):
        return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()