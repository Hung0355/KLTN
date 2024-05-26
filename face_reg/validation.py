import os
import dlib
import cv2
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Khởi tạo các mô hình Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Hàm để trích xuất face descriptors sử dụng Dlib
def get_face_descriptor(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_rgb, 1)
    
    if len(dets) > 0:
        shape = predictor(img_rgb, dets[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rgb, shape)
        return np.array(face_descriptor)
    else:
        return None

# Duyệt qua các thư mục và thu thập dữ liệu
data_dir = 'train_dataset/pack3'
image_paths = []
labels = []

for person_name in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person_name)
    if os.path.isdir(person_dir):
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image_paths.append(image_path)
            labels.append(person_name)

# Trích xuất các face descriptors và lưu trữ chúng
face_descriptors = []
valid_labels = []
for image_path, label in zip(image_paths, labels):
    descriptor = get_face_descriptor(image_path)
    if descriptor is not None:
        face_descriptors.append(descriptor)
        valid_labels.append(label)

face_descriptors = np.array(face_descriptors)
valid_labels = np.array(valid_labels)

# Mã hóa các nhãn thành số nguyên
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(valid_labels)


# Xây dựng mô hình phân loại sử dụng Keras
def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# Thiết lập K-Fold Cross-Validation
k = 5
kf = KFold(n_splits=k, random_state=42, shuffle=True)

accuracy_list = []
loss_list = []

for train_index, val_index in kf.split(face_descriptors):
    X_train, X_val = face_descriptors[train_index], face_descriptors[val_index]
    y_train, y_val = encoded_labels[train_index], encoded_labels[val_index]
    
    # Xây dựng và huấn luyện mô hình
    model = build_model(input_dim=X_train.shape[1], num_classes=len(np.unique(encoded_labels)))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    
    # Đánh giá mô hình trên tập validation
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    accuracy_list.append(accuracy)
    loss_list.append(loss)
    print(f"Accuracy for current fold: {accuracy}, Loss for current fold: {loss}")

# Tính độ chính xác và giá trị mất mát trung bình qua tất cả các fold
mean_accuracy = np.mean(accuracy_list)
mean_loss = np.mean(loss_list)
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Mean Loss: {mean_loss}")
