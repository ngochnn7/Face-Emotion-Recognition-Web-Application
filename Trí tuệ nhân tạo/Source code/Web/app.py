from flask import Flask, render_template, request, jsonify, Response
import cv2
import pickle
import os
import numpy as np
import threading
import time
import dlib

app = Flask(__name__)

models_config = {
    "naive_bayes": {
        "path": "models/Naive_Bayes",
        "model": None,
        "scaler": None,
        "k_best": None,
        "pca": None
    },
    "svm": {
        "path": "models/Support_Vector_Machine",
        "model": None,
        "scaler": None,
        "k_best": None,
        "pca": None
    },
    "random_forest": {
        "path": "models/Random_Forest",
        "model": None,
        "scaler": None,
        "k_best": None,
        "pca": None
    },
    "logistic_regression": {
        "path": "models/Logistic_Regression",
        "model": None,
        "scaler": None,
        "k_best": None,
        "pca": None
    },
    "knn": {
        "path": "models/K-Nearest_Neighbors",
        "model": None,
        "scaler": None,
        "k_best": None,
        "pca": None
    }
}

for model_name, config in models_config.items():
    model_path = config["path"]
    with open(os.path.join(model_path, "best_model.pkl"), "rb") as f:
        config["model"] = pickle.load(f)
    with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
        config["scaler"] = pickle.load(f)
    with open(os.path.join(model_path, "select_k_best.pkl"), "rb") as f:
        config["k_best"] = pickle.load(f)
    with open(os.path.join(model_path, "pca.pkl"), "rb") as f:
        config["pca"] = pickle.load(f)

classes = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Sadness", "Surprise"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = None
output_frame = None
lock = threading.Lock()
current_model = "svm"  

model_names = {
    "naive_bayes": "NB",
    "svm": "SVM",
    "random_forest": "RF",
    "logistic_regression": "LR",
    "knn": "KNN"
}

def preprocess_image(image_path, model_config):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Image not found or invalid format!")
        
        if len(img.shape) == 2: 
            gray = img
        else:  
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if gray.shape[0] == 48 and gray.shape[1] == 48:
            face_processed = gray
        else:
            max_dimension = 1200
            height, width = gray.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)
            
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                raise ValueError("No face detected in the image!")
            
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            margin = int(0.1 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(gray.shape[1] - x, w + 2 * margin)
            h = min(gray.shape[0] - y, h + 2 * margin)
            
            face = gray[y:y+h, x:x+w]
            
            face_processed = cv2.resize(face, (48, 48), interpolation=cv2.INTER_LANCZOS4)
        
        face_processed = cv2.normalize(face_processed, None, 0, 255, cv2.NORM_MINMAX)
        face_processed = cv2.GaussianBlur(face_processed, (5, 5), 0)
        face_processed = cv2.equalizeHist(face_processed)
        
        img_flattened = face_processed.flatten().reshape(1, -1)
        img_normalized = model_config["scaler"].transform(img_flattened)
        img_k_best = model_config["k_best"].transform(img_normalized)
        img_pca = model_config["pca"].transform(img_k_best)
        return img_pca
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def detect_emotion_camera(frame, model_config):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        emotions = []
        probabilities_list = []
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w] 
            try:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_processed = cv2.GaussianBlur(face_resized, (5, 5), 0)
                face_processed = cv2.equalizeHist(face_processed)
                
                img_flattened = face_processed.flatten().reshape(1, -1)
                img_normalized = model_config["scaler"].transform(img_flattened)
                img_k_best = model_config["k_best"].transform(img_normalized)
                img_pca = model_config["pca"].transform(img_k_best)
                
                probabilities = model_config["model"].predict_proba(img_pca)[0]
                predicted_index = np.argmax(probabilities)
                emotion = classes[predicted_index]
                
                emotions.append((emotion, x, y, w, h))
                probabilities_list.append(probabilities)
            except Exception as e:
                print(f"Lỗi xử lý khuôn mặt: {e}")
                continue
        
        for (emotion, x, y, w, h), probabilities in zip(emotions, probabilities_list):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            y_offset = 20
            for i, prob in enumerate(probabilities):
                if prob > 0:  
                    color = (0, 255, 0) if prob > 0.3 else (0, 255, 255) 
                    text = f"{classes[i]}: {prob*100:.2f}%"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
                    
        return frame
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return frame

def generate_frames():
    global output_frame, camera, current_model
    
    while True:
        if camera is None:
            break
            
        success, frame = camera.read()
        if not success:
            break
            
        model_config = models_config[current_model]
        processed_frame = detect_emotion_camera(frame, model_config)
        
        with lock:
            output_frame = processed_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("index.html")

@app.route("/emotion")
def emotion():
    return render_template("emotion.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "media" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    model_choice = request.form.get("model_choice")
    if model_choice not in models_config:
        return jsonify({"error": "Invalid model choice!"}), 400
    
    model_config = models_config[model_choice]
    media = request.files["media"]
    
    upload_dir = "static/uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    media_path = os.path.join(upload_dir, media.filename)
    media.save(media_path)

    try:
        if media.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            img_processed = preprocess_image(media_path, model_config)
            prediction = model_config["model"].predict(img_processed)
            predicted_emotion = classes[prediction[0]]
            probabilities = model_config["model"].predict_proba(img_processed)[0]  
            
            os.remove(media_path)
            emotion_probabilities = {classes[i]: prob for i, prob in enumerate(probabilities) if prob > 0}
            return jsonify({"emotion": predicted_emotion, "probabilities": emotion_probabilities})
        elif media.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(media_path, model_config)
            os.remove(media_path)
            return jsonify({"status": "Video processed successfully!"})
        else:
            os.remove(media_path)
            return jsonify({"error": "Unsupported file type!"}), 400
    except Exception as e:
        if os.path.exists(media_path):
            os.remove(media_path)
        return jsonify({"error": str(e)}), 500

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, current_model
    
    model_choice = request.form.get("model_choice")
    if model_choice in models_config:
        current_model = model_choice
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({"status": "success"})

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "success"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def preprocess_and_predict(face, model, scaler, k_best, pca, classes):
    pass

def process_video(video_path, model_config):
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Cannot open video!")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_width = 640 
    max_height = 480

    if frame_width > frame_height:
        scale_factor = max_width / frame_width
    else:
        scale_factor = max_height / frame_height

    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã đến cuối video hoặc không thể đọc frame!")
            break
        
        frame_resized = cv2.resize(frame, (new_width, new_height))

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        
        emotions = []
        probabilities_list = []
        
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_region = frame_resized[y:y+h, x:x+w]
            try:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_face, (48, 48))
                gray_resized = cv2.GaussianBlur(gray_resized, (5, 5), 0)
                gray_resized = cv2.equalizeHist(gray_resized)
                img_flattened = gray_resized.flatten().reshape(1, -1)
                img_normalized = model_config["scaler"].transform(img_flattened)
                img_k_best = model_config["k_best"].transform(img_normalized)
                img_pca = model_config["pca"].transform(img_k_best)
                
                probabilities = model_config["model"].predict_proba(img_pca)[0]
                predicted_index = np.argmax(probabilities)
                predicted_emotion = classes[predicted_index]
                
                emotions.append((predicted_emotion, x, y, w, h))
                probabilities_list.append(probabilities)
            except Exception as e:
                print(f"Lỗi xử lý khuôn mặt: {e}")
                continue
        
        for (emotion, x, y, w, h), probabilities in zip(emotions, probabilities_list):
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame_resized, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            y_offset = 20
            for i, prob in enumerate(probabilities):
                if prob > 0:
                    color = (0, 255, 0) if prob > 0.3 else (0, 255, 255)
                    text = f"{classes[i]}: {prob*100:.2f}%"
                    cv2.putText(frame_resized, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
        cv2.putText(frame_resized, f'Model: {model_names[current_model]}', (new_width // 2 - 100, new_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Emotion Detection", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    model_choice = request.form.get("model_choice")
    if model_choice not in models_config:
        return jsonify({"error": "Invalid model choice!"}), 400
    
    model_config = models_config[model_choice]  
    video = request.files["video"]
    
    upload_dir = "static/uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    video_path = os.path.join(upload_dir, video.filename)
    video.save(video_path)

    try:
        process_video(video_path, model_config)  
        os.remove(video_path)
        return jsonify({"status": "Video processed successfully!"})
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
