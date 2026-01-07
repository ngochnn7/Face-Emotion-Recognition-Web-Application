import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
import time

# Load dataset
data_directory = "/kaggle/input/ck-dataset/"
classes = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

images = []
labels = []

# Read images and labels
for idx, emotion in enumerate(classes):
    emotion_dir = os.path.join(data_directory, emotion)
    for file in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Preprocessing: Gaussian blur and histogram equalization
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.equalizeHist(img)
            images.append(img.flatten())
            labels.append(idx)

images = np.array(images)
labels = np.array(labels)

# Normalize data
scaler = StandardScaler()
images = scaler.fit_transform(images)

# Split the dataset into training (80%), validation (10%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Check class distribution before applying SMOTE
print("Class distribution before SMOTE:", Counter(y_train))

# Apply SMOTE to balance the training dataset
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Check class distribution after applying SMOTE
print("Class distribution after SMOTE:", Counter(y_train_res))

# K-Fold Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_k = 150
best_pca_components = 50
best_score = 0
best_var_smoothing = None

# Cross-validation to find optimal parameters for SelectKBest, PCA, and GaussianNB
for k in range(150, 300, 50):
    for n_components in range(50, 150, 50):
        for var_smoothing in [1e-9, 1e-8, 1e-7]:
            fold_scores = []

            for train_index, val_index in kf.split(X_train_res, y_train_res):
                X_train_fold, X_val_fold = X_train_res[train_index], X_train_res[val_index]
                y_train_fold, y_val_fold = y_train_res[train_index], y_train_res[val_index]

                # Feature selection with SelectKBest
                k_best = SelectKBest(f_classif, k=k)
                X_train_fold_k = k_best.fit_transform(X_train_fold, y_train_fold)
                X_val_fold_k = k_best.transform(X_val_fold)

                # Dimensionality reduction with PCA
                pca = PCA(n_components=n_components)
                X_train_fold_pca = pca.fit_transform(X_train_fold_k)
                X_val_fold_pca = pca.transform(X_val_fold_k)

                # Model training
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(X_train_fold_pca, y_train_fold)

                # Cross-validation score on validation fold
                score = model.score(X_val_fold_pca, y_val_fold)
                fold_scores.append(score)

            # Calculate average score for this combination
            avg_score = np.mean(fold_scores)

            # Update the best parameters if the average score improves
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
                best_pca_components = n_components
                best_var_smoothing = var_smoothing

print(f"Best k: {best_k}")
print(f"Best PCA components: {best_pca_components}")
print(f"Best GaussianNB var_smoothing: {best_var_smoothing}")

# Feature selection with the best k
k_best = SelectKBest(f_classif, k=best_k)
X_train_res_k = k_best.fit_transform(X_train_res, y_train_res)
X_val_k = k_best.transform(X_val)
X_test_k = k_best.transform(X_test)

# Dimensionality reduction with the best PCA components
pca = PCA(n_components=best_pca_components)
X_train_res_pca = pca.fit_transform(X_train_res_k)
X_val_pca = pca.transform(X_val_k)
X_test_pca = pca.transform(X_test_k)

# Train the model with the best configuration
model = GaussianNB(var_smoothing=best_var_smoothing)

# Measure the time taken to train the model
start_time = time.time()
model.fit(X_train_res_pca, y_train_res)
training_time = time.time() - start_time

# Evaluate the model on train, validation, and test sets
print("Train Classification Report:")
print(classification_report(y_train_res, model.predict(X_train_res_pca), target_names=classes))

print("Validation Classification Report:")
print(classification_report(y_val, model.predict(X_val_pca), target_names=classes))

print("Test Classification Report:")
print(classification_report(y_test, model.predict(X_test_pca), target_names=classes))

# Confusion matrix on the test set
y_test_pred = model.predict(X_test_pca)
cm = confusion_matrix(y_test, y_test_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.show()

# Calculate and print Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Calculate and print Log-Loss
logloss = log_loss(y_test, model.predict_proba(X_test_pca))
print(f"Log-loss: {logloss:.4f}")

# Calculate and print ROC-AUC score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_pca), multi_class='ovr')
print(f"ROC-AUC: {roc_auc:.2f}")

# Print training time
print(f'Training Time: {training_time:.4f} seconds')

# Save the best model, SelectKBest, PCA, and Scaler
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("select_k_best.pkl", "wb") as f:
    pickle.dump(k_best, f)

with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
