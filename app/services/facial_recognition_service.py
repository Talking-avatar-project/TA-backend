# services/facial_recognition_service.py
from app.utils.emotion_recognition import EmotionRecognition
from app.utils.media_pipe_utils import process_image_with_mediapipe, batch_process_images  # Asegúrate de importar batch_process_images

class FacialRecognitionService:
    @staticmethod
    def process_face_data(frame):
        # Instancia del modelo de reconocimiento de emociones
        emotion_recognition = EmotionRecognition()
        # Procesa el frame para detectar la emoción
        response = emotion_recognition.detect_emotion(frame)

        # Retorna el resultado de la emoción detectada
        return response

    @staticmethod  # Añade @staticmethod si la función es estática
    def analyze_images_for_avatar():
        input_folder = 'images/raw_images/'
        output_folder = 'images/processed_images/'

        # Procesar imágenes en lote
        batch_process_images(input_folder, output_folder)
        return "Images processed with facial landmarks saved."
