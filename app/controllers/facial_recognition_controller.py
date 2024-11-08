# controllers/facial_recognition_controller.py

from flask import Blueprint, request, jsonify
from app.utils.emotion_recognition import EmotionRecognition
import base64
import numpy as np
import cv2

# Crear Blueprint para el reconocimiento facial
facial_recognition_bp = Blueprint('facial_recognition', __name__)

# Crear una instancia de la clase de reconocimiento de emociones
emotion_recognition = EmotionRecognition()


@facial_recognition_bp.route('/recognize', methods=['POST'])
def recognize():
    # Recibe el JSON de la solicitud
    data = request.get_json()
    face_data = data.get("face_data")

    # Validar que se proporcionó la imagen en base64
    if not face_data:
        return jsonify({"error": "No face data provided"}), 400

    # Decodificar la imagen de base64 a un formato de numpy array que OpenCV puede procesar
    try:
        # Eliminar el encabezado del string base64 si lo incluye (ej. data:image/jpeg;base64,)
        if "base64," in face_data:
            face_data = face_data.split("base64,")[1]

        # Decodificar base64 a bytes
        image_bytes = base64.b64decode(face_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decodificar bytes a imagen
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": "Failed to process image data"}), 400

    # Detectar emoción en la imagen
    response = emotion_recognition.detect_emotion(image)

    # Retornar el resultado en formato JSON
    return jsonify(response)
