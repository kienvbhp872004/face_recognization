import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Lớp tải và xử lý dữ liệu khuôn mặt
class FaceLoading:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (112, 112)
        self.X = []
        self.y = []
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
        self.i = 1

    def extract_face(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = self.mtcnn(img)
        return face

    def load_face_and_class(self):
        for sub_dir in os.listdir(self.directory):
            sub_dir_path = os.path.join(self.directory, sub_dir)
            for img_name in os.listdir(sub_dir_path):
                face = self.extract_face(os.path.join(sub_dir_path, img_name))
                if (face is not None):
                    self.X.append(face)
                    self.y.append(sub_dir)
        return np.array(self.X), np.array(self.y)

    def plot_images(self):
        num_columns = 3
        num_rows = math.ceil(len(self.X) / num_columns)  # Ensure row count covers all images

        plt.figure(figsize=(num_columns * 3, num_rows * 3))  # Adjust figure size dynamically
        for num, img in enumerate(self.X):
            plt.subplot(num_rows, num_columns, num + 1)
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.axis('off')
        plt.tight_layout()  # Improve layout spacing
        plt.show()


# Hàm tiền xử lý ảnh
TEMPLATE = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)


def preprocess_image(image_path, image_size=(112, 112)):
    try:
        # Đọc và chuyển đổi ảnh
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Khởi tạo MTCNN
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                      thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                      device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        # Phát hiện khuôn mặt và landmarks
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        if boxes is None or len(boxes) == 0:
            raise ValueError(f"No face detected in {image_path}")

        # Chuyển đổi PIL Image về numpy array để căn chỉnh
        img_array = np.array(img)

        # Căn chỉnh khuôn mặt nếu có landmarks
        if landmarks is not None and len(landmarks) > 0:
            landmark = landmarks[0]
            if landmark is not None and len(landmark) == 5:
                # Căn chỉnh khuôn mặt
                src = np.array(landmark).astype(np.float32)
                M = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)[0]
                aligned_face = cv2.warpAffine(img_array, M, (112, 112), borderValue=0.0)

                # Chuẩn hóa dữ liệu
                face = aligned_face.astype(np.float32)
                face = (face - 127.5) / 128.0  # Chuẩn hóa về [-1, 1]
                face = np.expand_dims(face, axis=0)  # shape: (1, 112, 112, 3)
                return face

        # Fallback: sử dụng phương pháp cũ nếu không căn chỉnh được
        face_tensor = mtcnn(img)  # shape: (3, 160, 160)
        if face_tensor is None:
            raise ValueError(f"No face detected in {image_path}")

        face = face_tensor.permute(1, 2, 0).cpu().numpy()  # shape: (160, 160, 3)
        face = cv2.resize(face, image_size)  # resize về (112, 112, 3)
        face = (face - 0.5) * 2.0  # chuẩn hóa [-1, 1]
        face = np.expand_dims(face, axis=0)  # shape: (1, 112, 112, 3)
        return face

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None



# Định nghĩa lớp ArcFaceLoss
class ArcFaceLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, margin=0.5, scale=30.0, embedding_size=512,
                 name='arcface_loss', reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        super(ArcFaceLoss, self).__init__(name=name, reduction=reduction)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.embedding_size = embedding_size
        self.threshold = math.cos(math.pi - margin)
        self.cos_m = tf.cos(margin)
        self.sin_m = tf.sin(margin)
        self.W = tf.Variable(tf.random.normal([embedding_size, num_classes], stddev=0.1),
                             trainable=True, name='arcface_weights')

    def call(self, y_true, y_pred):
        embeddings = tf.nn.l2_normalize(y_pred, axis=1)
        weights = tf.nn.l2_normalize(self.W, axis=0)
        cos_theta = tf.matmul(embeddings, weights)
        theta = tf.acos(tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = tf.cos(theta + self.margin)
        logits = cos_theta * (1.0 - y_true) + target_logits * y_true
        logits = logits * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=True)


# Load lại mô hình
def load_arcface_model(model_path, num_classes=1252):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'ArcFaceLoss': ArcFaceLoss(num_classes)}
    )
    return model


model = tf.keras.models.load_model(
    r'C:\Users\admin\OneDrive - Hanoi University of Science and Technology\Documents\GitHub\PTTK\face_recognization\model\arcface_model_final.h5',
    custom_objects={'ArcFaceLoss': lambda: ArcFaceLoss(num_classes=1252)}
)

# model.summary()


def create_embedding_database(model, X, y):
    embedding_db = {}
    unique_labels = np.unique(y)

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        label_embeddings = []

        for idx in label_indices:
            face_tensor = X[idx]
            if isinstance(face_tensor, torch.Tensor):
                face = face_tensor.permute(1, 2, 0).cpu().numpy()
                print(face.shape())# CHW → HWC
            else:
                face = face_tensor  # đã là numpy rồi

            # Resize và chuẩn hóa về [-1, 1] nếu chưa
            if face.shape != (112, 112, 3):
                face = cv2.resize(face, (112, 112))
            face = (face - 0.5) * 2.0
            face = np.expand_dims(face, axis=0)  # (1, 112, 112, 3)

            embedding = model.predict(face, verbose=0)
            embedding = tf.nn.l2_normalize(embedding, axis=1).numpy()[0]
            label_embeddings.append(embedding)

        # Trung bình embedding của mỗi người
        embedding_db[label] = np.mean(label_embeddings, axis=0)
    return embedding_db


# Hàm tính cosine similarity
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# Hàm phân loại ảnh mới
def classify_face(model, image_path, embedding_db, threshold=0.8):
    image = preprocess_image(image_path)
    if image is None:
        return "Invalid image", -1

    embedding = model.predict(image, verbose=0)
    embedding = tf.nn.l2_normalize(embedding, axis=1).numpy()[0]

    max_similarity = -1
    predicted_label = None
    for label, db_embedding in embedding_db.items():
        similarity = cosine_similarity(embedding, db_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_label = label

    if max_similarity >= threshold:
        return predicted_label, max_similarity
    return "Unknown", max_similarity


# Main execution
# Tải dữ liệu khuôn mặt
face_loading = FaceLoading(
    r"C:\Users\admin\OneDrive - Hanoi University of Science and Technology\Documents\GitHub\PTTK\face_recognization\source\data_raw\image")
X, y = face_loading.load_face_and_class()
print(f"Loaded {len(X)} faces with {len(np.unique(y))} unique identities")

# Mã hóa nhãn
encode = LabelEncoder()
encoded_y = encode.fit_transform(y)
encoded_y = encoded_y.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_y = onehot_encoder.fit_transform(encoded_y)
print("One-hot encoded labels shape:", onehot_y.shape)

# Tạo cơ sở dữ liệu embeddings
embedding_db = create_embedding_database(model, X, y)
np.save('embedding_database.npy', embedding_db)
print("Embedding database created and saved")



