# utils/mobilenet_emotion_model.py
import cv2
import tensorflow as tf
import numpy as np


class MobileNetEmotionModel:
    def __init__(self):
        # Carga el modelo MobileNet preentrenado (ajústalo según sea necesario)
        self.model = tf.keras.applications.MobileNetV2(weights="imagenet")

    def preprocess_input(self, frame):
        # Preprocesa el frame de la cámara para que sea compatible con MobileNet
        resized_frame = cv2.resize(frame, (224, 224))
        img_array = tf.keras.applications.mobilenet.preprocess_input(resized_frame)
        return np.expand_dims(img_array, axis=0)

    def predict_emotion(self, frame):
        processed_frame = self.preprocess_input(frame)
        predictions = self.model.predict(processed_frame)
        # Aquí debes convertir las predicciones a emociones
        # Este paso requiere un modelo MobileNet especializado en emociones
        # Por ahora, devolveremos la clase predicha como prueba
        return tf.keras.applications.mobilenet.decode_predictions(predictions, top=1)[0][0]
