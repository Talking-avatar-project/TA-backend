# controllers/facial_recognition_controller.py
from flask import Blueprint, jsonify
from app.utils.emotion_recognition import EmotionRecognition
import cv2

facial_recognition_bp = Blueprint('facial_recognition', __name__)
emotion_recognition = EmotionRecognition()

@facial_recognition_bp.route('/facial_recognition/stream', methods=['GET'])
def stream():
    # Abrir cámara
    cap = cv2.VideoCapture(0)  # Cámara integrada
    if not cap.isOpened():
        return jsonify({"error": "No se pudo acceder a la cámara"}), 500

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada fotograma
            response = emotion_recognition.detect_emotion(frame)
            print(response)  # Mostrar emoción detectada en consola (temporal)

        return jsonify({"message": "Streaming finalizado"})
    finally:
        cap.release()
