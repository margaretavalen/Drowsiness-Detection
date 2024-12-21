import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model CNN
MODEL_PATH = "fcnn_model.h5"  # Ganti dengan model Anda
model = load_model(MODEL_PATH)

# Threshold dan konfigurasi deteksi
EYE_CLOSED_THRESHOLD = 0.5  # Probabilitas threshold untuk mata tertutup
CLOSED_FRAMES_THRESHOLD = 30  # Jumlah frame mata tertutup untuk mendeteksi kantuk

# Fungsi untuk memproses gambar mata
def preprocess_eye(eye_image):
    eye_image = cv2.resize(eye_image, (64, 64))  # Resize sesuai kebutuhan
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale
    eye_image = eye_image.astype("float") / 255.0  # Normalisasi
    eye_image = eye_image.flatten()  # Ratakan array menjadi 1D
    eye_image = np.expand_dims(eye_image, axis=0)  # Tambahkan batch dimension
    return eye_image

# Streamlit UI
st.title("Drowsiness Detection with CNN")
st.text("Real-time video feed to detect drowsiness.")

run = st.checkbox("Start Detection")

if run:
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)  # Menggunakan webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    closed_frames = 0
    drowsy_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face)

            for (ex, ey, ew, eh) in eyes:
                eye = face[ey:ey + eh, ex:ex + ew]
                processed_eye = preprocess_eye(eye)
                prediction = model.predict(processed_eye)
                prob_closed = prediction[0][0]  # Probabilitas mata tertutup

                # Deteksi kantuk berdasarkan probabilitas
                if prob_closed > EYE_CLOSED_THRESHOLD:
                    closed_frames += 1
                    color = (0, 0, 255)  # Merah jika mata tertutup
                    status = "Closed"
                else:
                    closed_frames = 0
                    color = (0, 255, 0)  # Hijau jika mata terbuka
                    status = "Open"

                # Gambar kotak di sekitar mata dan teks hasil deteksi
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), color, 2)
                cv2.putText(face, f"{status}: {prob_closed:.2f}", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Tambahkan peringatan jika mata tertutup terlalu lama
        if closed_frames > CLOSED_FRAMES_THRESHOLD:
            drowsy_detected = True
            cv2.putText(frame, "Drowsiness Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            drowsy_detected = False

        # Tampilkan frame video
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.write("Click the checkbox to start detection.")
