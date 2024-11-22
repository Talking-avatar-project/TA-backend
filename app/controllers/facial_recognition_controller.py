# facial_recognition_controller.py
from flask import Blueprint, jsonify, request
from app.services.facial_recognition_service import FacialRecognitionService  # Solo importa la clase
import cv2
import base64
import numpy as np

facial_recognition_bp = Blueprint('facial_recognition', __name__)

# Endpoint para iniciar el análisis de imágenes para el avatar
@facial_recognition_bp.route('/process-avatar-images', methods=['GET'])
def process_avatar_images():
    result = FacialRecognitionService.analyze_images_for_avatar()
    return jsonify({"message": result})

# Endpoint para streaming en tiempo real
@facial_recognition_bp.route('/stream', methods=['GET'])
def stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "No se pudo acceder a la cámara"}), 500

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada fotograma
            response = FacialRecognitionService.process_face_data(frame)
            print(response)  # Mostrar emoción detectada en consola (temporal)

        return jsonify({"message": "Streaming finalizado"})
    finally:
        cap.release()

# Endpoint para reconocimiento de imágenes enviadas por POST
@facial_recognition_bp.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    face_data = data.get("face_data")

    if not face_data:
        return jsonify({"error": "No face data provided"}), 400

    # Decodifica y procesa la imagen
    image_bytes = base64.b64decode(face_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    response = FacialRecognitionService.process_face_data(image)
    return jsonify(response)
