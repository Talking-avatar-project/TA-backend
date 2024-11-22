# utils/emotion_recognition.py
from fer import FER
import cv2
import numpy as np

class EmotionRecognition:
    def __init__(self):
        # Carga del modelo FER preentrenado con MTCNN activado
        self.detector = FER(mtcnn=True)

    def preprocess_image(self, image):
        # Convertir a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Ajustar brillo y contraste
        adjusted_image = cv2.convertScaleAbs(gray_image, alpha=1.5, beta=20)  # Parámetros ajustables
        # Volver a convertir a BGR para mantener compatibilidad con MTCNN
        return cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)

    def detect_emotion(self, image):
        # Preprocesar imagen
        preprocessed_image = self.preprocess_image(image)
        # Detectar emociones en la imagen preprocesada
        results = self.detector.top_emotion(preprocessed_image)

        if results:
            emotion, score = results
            return {
                "detected_emotion": emotion,
                "confidence": score
            }
        else:
            return {
                "error": "No face detected"
            }
