import cv2
import os
import time


class AvatarService:
    @staticmethod
    def animate_avatar(emotion):
        # Definir ruta a las imágenes procesadas con landmarks
        image_folder = 'images/processed_images'

        # Mapear emociones a conjuntos de imágenes
        emotion_images = {
            "neutral": ["base_01_var_01.jpg", "base_01_var_02.jpg", ...],
            "feliz": ["base_02_var_01.jpg", "base_02_var_02.jpg", ...],
            "sorprendido": ["base_03_var_01.jpg", "base_03_var_02.jpg", ...]
            # Agregar más según las emociones disponibles
        }

        # Seleccionar imágenes correspondientes a la emoción detectada
        selected_images = emotion_images.get(emotion, emotion_images["neutral"])

        # Mostrar cada imagen en secuencia para crear la animación
        for img_name in selected_images:
            img_path = os.path.join(image_folder, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                cv2.imshow("Avatar Animation", image)
                # Pausa para crear efecto de animación (ajusta el tiempo según necesidad)
                cv2.waitKey(100)  # 100 ms entre imágenes para suavidad

        cv2.destroyAllWindows()
