# utils/emotion_recognition.py
from fer import FER
import cv2

class EmotionRecognition:
    def __init__(self):
        # Carga el modelo FER preentrenado
        self.detector = FER(mtcnn=True)

    def detect_emotion(self, image):
        """
        Detecta la emoción en la imagen proporcionada.
        :param image: Imagen (fotograma) en formato numpy array (capturada con OpenCV).
        :return: Diccionario con la emoción detectada y su confianza.
        """
        # Detecta emociones en la imagen
        results = self.detector.top_emotion(image)
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
