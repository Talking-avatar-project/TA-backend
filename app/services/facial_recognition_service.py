# services/facial_recognition_service.py
from app.utils.emotion_recognition import EmotionRecognition


class FacialRecognitionService:
    @staticmethod
    def process_face_data(frame):
        # Instancia del modelo de reconocimiento de emociones
        emotion_recognition = EmotionRecognition()
        # Procesa el frame para detectar la emoción
        response = emotion_recognition.detect_emotion(frame)

        # Retorna el resultado de la emoción detectada
        return response
