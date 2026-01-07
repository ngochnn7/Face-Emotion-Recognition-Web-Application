import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
import time

# Tải tập dữ liệu
data_directory = "/kaggle/input/ck-dataset/"
classes = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

images = []
labels = []

# Đọc hình ảnh và nhãn
for idx, emotion in enumerate(classes):
    emotion_dir = os.path.join(data_directory, emotion)
    for file in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Tiền xử lý: Làm mờ Gauss và cân bằng biểu đồ
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.equalizeHist(img)
            images.append(img.flatten())
            labels.append(idx)

images = np.array(images)
labels = np.array(labels)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
images = scaler.fit_transform(images)

# Chia tập dữ liệu thành các tập huấn luyện 80%, xác thực 10% và kiểm tra 10%
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Kiểm tra phân phối lớp trước khi áp dụng SMOTE
print("Class distribution before SMOTE:", Counter(y_train))

# Áp dụng SMOTE để cân bằng bộ dữ liệu đào tạo
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Kiểm tra phân phối lớp sau SMOTE
print("Class distribution after SMOTE:", Counter(y_train_res))

# K-Fold Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Các tham số để thử nghiệm
kernel_options = ['linear', 'rbf', 'poly']
C_values = [0.1, 1, 10]

best_params = {}
best_score = 0

# Tìm tham số tối ưu
for kernel in kernel_options:
    for C in C_values:
        for k in range(150, 300, 50):
            for n_components in range(50, 150, 50):
                fold_scores = []

                for train_index, val_index in kf.split(X_train_res, y_train_res):
                    X_train_fold, X_val_fold = X_train_res[train_index], X_train_res[val_index]
                    y_train_fold, y_val_fold = y_train_res[train_index], y_train_res[val_index]

                    # Lựa chọn tính năng với SelectKBest
                    k_best = SelectKBest(f_classif, k=k)
                    X_train_fold_k = k_best.fit_transform(X_train_fold, y_train_fold)
                    X_val_fold_k = k_best.transform(X_val_fold)

                    # Giảm chiều với PCA
                    pca = PCA(n_components=n_components)
                    X_train_fold_pca = pca.fit_transform(X_train_fold_k)
                    X_val_fold_pca = pca.transform(X_val_fold_k)

                    # Huấn luyện mô hình
                    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
                    model.fit(X_train_fold_pca, y_train_fold)

                    # Đánh giá trên tập xác thực
                    score = model.score(X_val_fold_pca, y_val_fold)
                    fold_scores.append(score)

                # Tính điểm trung bình
                avg_score = np.mean(fold_scores)

                # Cập nhật tham số tốt nhất nếu điểm trung bình được cải thiện
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        'kernel': kernel,
                        'C': C,
                        'k': k,
                        'n_components': n_components
                    }

print("Best parameters:", best_params)
print(f"Best cross-validation score: {best_score:.4f}")

# Lựa chọn tham số tốt nhất
k_best = SelectKBest(f_classif, k=best_params['k'])
X_train_res_k = k_best.fit_transform(X_train_res, y_train_res)
X_val_k = k_best.transform(X_val)
X_test_k = k_best.transform(X_test)

pca = PCA(n_components=best_params['n_components'])
X_train_res_pca = pca.fit_transform(X_train_res_k)
X_val_pca = pca.transform(X_val_k)
X_test_pca = pca.transform(X_test_k)

# Huấn luyện mô hình tốt nhất
model = SVC(kernel=best_params['kernel'], C=best_params['C'], probability=True, random_state=42)
start_time = time.time()
model.fit(X_train_res_pca, y_train_res)
training_time = time.time() - start_time

# Đánh giá mô hình
print("Train Classification Report:")
print(classification_report(y_train_res, model.predict(X_train_res_pca), target_names=classes))

print("Validation Classification Report:")
print(classification_report(y_val, model.predict(X_val_pca), target_names=classes))

print("Test Classification Report:")
print(classification_report(y_test, model.predict(X_test_pca), target_names=classes))

# Confusion matrix
y_test_pred = model.predict(X_test_pca)
cm = confusion_matrix(y_test, y_test_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.show()

# Tính và in Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Tính và in Log-Loss
logloss = log_loss(y_test, model.predict_proba(X_test_pca))
print(f"Log-loss: {logloss:.4f}")

# Tính và in ROC-AUC score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_pca), multi_class='ovr')
print(f"ROC-AUC: {roc_auc:.2f}")

# In thời gian huấn luyện
print(f'Training Time: {training_time:.4f} seconds')

# Lưu mô hình tốt nhất, SelectKBest, PCA, và Scaler
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("select_k_best.pkl", "wb") as f:
    pickle.dump(k_best, f)

with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
