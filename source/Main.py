import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model


class FaceRecognitionSystem:
    def __init__(self, model_type='arcface',
                 facenet_embeddings_path='face_embeddings.npy',
                 facenet_labels_path='face_labels.npy',
                 resnet_embeddings_path='user_embeddings.npy',
                 resnet_labels_path='user_labels.npy',
                 resnet_model_path=None):

        self.model_type = model_type
        print(f"Khởi tạo hệ thống với model: {model_type}")

        if model_type == 'arcface':
            self._init_facenet_model()
            self.load_known_faces(facenet_embeddings_path, facenet_labels_path)
        elif model_type == 'resnet':
            self._init_resnet_model(resnet_model_path)
            self.load_known_faces(resnet_embeddings_path, resnet_labels_path)
        else:
            raise ValueError("model_type phải là 'arcface' hoặc 'resnet'")

        # Threshold cho cosine similarity
        self.threshold = 0.7 if model_type == 'arcface' else 0.6

    def _init_facenet_model(self):
        """Khởi tạo FaceNet model"""
        # Khởi tạo device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Khởi tạo MTCNN để detect face
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )

        # Khởi tạo InceptionResnetV1 để extract features
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _init_resnet_model(self, model_path):
        """Khởi tạo ResNet50 model"""
        if model_path is None:
            raise ValueError("Cần cung cấp đường dẫn model ResNet50")

        # Load ResNet50 model
        self.resnet_model = load_model(model_path)
        self.embedding_model = Model(inputs=self.resnet_model.input,
                                     outputs=self.resnet_model.get_layer('flatten_1').output)

        # Khởi tạo face detector (sử dụng Haar Cascade cho đơn giản)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.target_size = (112, 112)

    def load_known_faces(self, embeddings_path, labels_path):
        """Load embeddings và labels đã lưu"""
        try:
            self.known_embeddings = np.load(embeddings_path)
            self.known_labels = np.load(labels_path)
            print(f"Đã load {len(self.known_embeddings)} embeddings với {len(set(self.known_labels))} người khác nhau")
            print(f"Các người trong database: {set(self.known_labels)}")
        except FileNotFoundError:
            print("Không tìm thấy file embeddings hoặc labels!")
            print("Vui lòng chạy script tạo embeddings trước")
            self.known_embeddings = np.array([])
            self.known_labels = np.array([])

    def extract_face_embedding_facenet(self, face_tensor):
        """Extract embedding từ face tensor sử dụng FaceNet"""
        with torch.no_grad():
            embedding = self.resnet(face_tensor).cpu().detach().numpy()
        return embedding.flatten()

    def extract_face_embedding_resnet(self, face_image):
        """Extract embedding từ face image sử dụng ResNet50"""
        # Resize và normalize
        face_resized = cv2.resize(face_image, self.target_size)
        face_norm = face_resized / 255.0

        # Predict embedding
        embedding = self.embedding_model.predict(face_norm[np.newaxis], verbose=0)[0]
        return embedding

    def recognize_face(self, face_embedding):
        """Nhận diện khuôn mặt dựa trên embedding"""
        if len(self.known_embeddings) == 0:
            return "Unknown", 0.0

        # Tính cosine similarity với tất cả embeddings đã biết
        similarities = cosine_similarity([face_embedding], self.known_embeddings)[0]

        # Tìm similarity cao nhất
        max_similarity = np.max(similarities)
        best_match_idx = np.argmax(similarities)

        # Kiểm tra threshold
        if max_similarity >= self.threshold:
            predicted_label = self.known_labels[best_match_idx]
            return predicted_label, max_similarity
        else:
            return "Unknown", max_similarity

    def process_frame_facenet(self, frame):
        """Xử lý frame sử dụng FaceNet"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = self.mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                # Extract face region
                x1, y1, x2, y2 = box.astype(int)

                # Đảm bảo tọa độ trong phạm vi frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Extract face từ frame
                face_region = rgb_frame[y1:y2, x1:x2]

                if face_region.size > 0:
                    # Convert to PIL Image
                    face_pil = Image.fromarray(face_region)

                    # Preprocess
                    face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)

                    # Extract embedding
                    embedding = self.extract_face_embedding_facenet(face_tensor)

                    # Recognize
                    name, confidence = self.recognize_face(embedding)

                    # Draw bounding box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{name} ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def process_frame_resnet(self, frame):
        """Xử lý frame sử dụng ResNet50"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Extract face region
            face_region = frame[y:y + h, x:x + w]

            if face_region.size > 0:
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                # Extract embedding
                embedding = self.extract_face_embedding_resnet(face_rgb)

                # Recognize
                name, confidence = self.recognize_face(embedding)

                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw label
                label = f"{name} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - 25), (x + label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def process_frame(self, frame):
        """Xử lý frame theo loại model"""
        if self.model_type == 'arcface':
            return self.process_frame_facenet(frame)
        else:
            return self.process_frame_resnet(frame)

    def run_camera(self):
        """Chạy camera và nhận diện real-time"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Không thể mở camera")
            return

        print(f"Camera đã khởi động với model {self.model_type}. Nhấn 'q' để thoát, 's' để chụp ảnh")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể nhận frame")
                break

            # Process frame
            processed_frame = self.process_frame(frame)

            # Hiển thị frame
            cv2.imshow(f"Face Recognition - {self.model_type}", processed_frame)

            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Lưu ảnh
                cv2.imwrite(f'captured_frame_{self.model_type}_{np.random.randint(1000, 9999)}.jpg', processed_frame)
                print("Đã lưu ảnh")

        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()

    def test_image(self, image_path):
        """Test với một ảnh cụ thể"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Không thể đọc ảnh: {image_path}")
            return

        processed_frame = self.process_frame(frame)

        # Hiển thị kết quả
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Face Recognition Result - {self.model_type}")
        plt.axis('off')
        plt.show()

    def add_new_person_facenet(self, name, num_photos=5):
        """Thêm người mới vào database sử dụng FaceNet"""
        print(f"Thêm người mới: {name}")
        print(f"Sẽ chụp {num_photos} ảnh. Nhấn 'c' để chụp, 'q' để thoát")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera")
            return

        captured_embeddings = []
        captured_count = 0

        while captured_count < num_photos:
            ret, frame = cap.read()
            if not ret:
                break

            # Hiển thị frame
            cv2.putText(frame, f"Captured: {captured_count}/{num_photos}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Add New Person - ArcFace", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Xử lý frame để extract embedding
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(rgb_frame)

                if boxes is not None and len(boxes) > 0:
                    # Lấy face đầu tiên
                    box = boxes[0]
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    face_region = rgb_frame[y1:y2, x1:x2]
                    if face_region.size > 0:
                        face_pil = Image.fromarray(face_region)
                        face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)
                        embedding = self.extract_face_embedding_facenet(face_tensor)
                        captured_embeddings.append(embedding)
                        captured_count += 1
                        print(f"Đã chụp {captured_count}/{num_photos}")
                else:
                    print("Không detect được face, thử lại")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_embeddings:
            self._save_new_embeddings(captured_embeddings, name, 'face_embeddings.npy', 'face_labels.npy')

    def add_new_person_resnet(self, name, num_photos=5):
        """Thêm người mới vào database sử dụng ResNet50"""
        print(f"Thêm người mới: {name}")
        print(f"Sẽ chụp {num_photos} ảnh. Nhấn 'c' để chụp, 'q' để thoát")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera")
            return

        captured_embeddings = []
        captured_count = 0

        while captured_count < num_photos:
            ret, frame = cap.read()
            if not ret:
                break

            # Hiển thị frame
            cv2.putText(frame, f"Captured: {captured_count}/{num_photos}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Add New Person - ResNet", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    # Lấy face đầu tiên
                    x, y, w, h = faces[0]
                    face_region = frame[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

                    # Extract embedding
                    embedding = self.extract_face_embedding_resnet(face_rgb)
                    captured_embeddings.append(embedding)
                    captured_count += 1
                    print(f"Đã chụp {captured_count}/{num_photos}")
                else:
                    print("Không detect được face, thử lại")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_embeddings:
            self._save_new_embeddings(captured_embeddings, name, 'user_embeddings.npy', 'user_labels.npy')

    def add_new_person(self, name, num_photos=5):
        """Thêm người mới vào database"""
        if self.model_type == 'arcface':
            self.add_new_person_facenet(name, num_photos)
        else:
            self.add_new_person_resnet(name, num_photos)

    def _save_new_embeddings(self, captured_embeddings, name, embeddings_file, labels_file):
        """Lưu embeddings mới vào file"""
        new_embeddings = np.array(captured_embeddings)
        new_labels = np.array([name] * len(captured_embeddings))

        if len(self.known_embeddings) > 0:
            self.known_embeddings = np.vstack([self.known_embeddings, new_embeddings])
            self.known_labels = np.hstack([self.known_labels, new_labels])
        else:
            self.known_embeddings = new_embeddings
            self.known_labels = new_labels

        # Lưu lại
        np.save(embeddings_file, self.known_embeddings)
        np.save(labels_file, self.known_labels)
        print(f"Đã thêm {len(captured_embeddings)} embeddings cho {name}")


# Sử dụng hệ thống
if __name__ == "__main__":
    print("=== HỆ THỐNG NHẬN DIỆN KHUÔN MẶT ===")
    print("Chọn loại model:")
    print("1. ArcFace (PyTorch)")
    print("2. ResNet50 (TensorFlow)")

    model_choice = input("Chọn model (1 hoặc 2): ")

    if model_choice == '1':
        face_system = FaceRecognitionSystem(model_type='arcface')
    elif model_choice == '2':
        resnet_model_path = r"C:\Users\admin\OneDrive - Hanoi University of Science and Technology\Documents\GitHub\PTTK\face_recognization\model\resnet50.h5"
        face_system = FaceRecognitionSystem(
            model_type='resnet',
            resnet_model_path=resnet_model_path
        )
    else:
        print("Lựa chọn không hợp lệ")
        exit()

    print("\n=== MENU CHÍNH ===")
    print("1. Chạy camera nhận diện")
    print("2. Test với ảnh")
    print("3. Thêm người mới")
    print("4. Thoát")

    while True:
        choice = input("\nChọn chức năng (1-4): ")

        if choice == '1':
            face_system.run_camera()
        elif choice == '2':
            image_path = input("Nhập đường dẫn ảnh: ")
            face_system.test_image(image_path)
        elif choice == '3':
            name = input("Nhập tên người mới: ")
            num_photos = int(input("Số ảnh cần chụp (mặc định 5): ") or "5")
            face_system.add_new_person(name, num_photos)
        elif choice == '4':
            print("Thoát chương trình")
            break
        else:
            print("Lựa chọn không hợp lệ")