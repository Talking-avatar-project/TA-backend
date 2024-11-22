# real_time_emotion_detection.py
import sys
import os
import threading
import time
import cv2
from fer import FER

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

test_image = cv2.imread("../..images/processed_images/base_02_var_11.jpg")
if test_image is None:
    print("La imagen no se encuentra en la ruta especificada.")
else:
    print("Imagen cargada correctamente.")

IMAGE_PATHS = {
    "neutral": [
        "../../images/processed_images/base_02_var_11.jpg", "../../images/processed_images/base_02_var_12.jpg",
        "../../images/processed_images/base_02_var_13.jpg", "../../images/processed_images/base_02_var_14.jpg",
        "../../images/processed_images/base_02_var_15.jpg", "../../images/processed_images/base_02_var_16.jpg",
        "../../images/processed_images/base_02_var_17.jpg", "../../images/processed_images/base_02_var_18.jpg",
        "../../images/processed_images/base_02_var_19.jpg", "../../images/processed_images/base_02_var_20.jpg",
        "../../images/processed_images/base_03_var_21.jpg", "../../images/processed_images/base_03_var_22.jpg",
        "../../images/processed_images/base_03_var_23.jpg", "../../images/processed_images/base_03_var_24.jpg",
        "../../images/processed_images/base_03_var_25.jpg", "../../images/processed_images/base_03_var_26.jpg",
        "../../images/processed_images/base_03_var_27.jpg", "../../images/processed_images/base_03_var_28.jpg",
        "../../images/processed_images/base_03_var_29.jpg", "../../images/processed_images/base_03_var_30.jpg",
        "../../images/processed_images/base_04_var_31.jpg", "../../images/processed_images/base_04_var_32.jpg",
        "../../images/processed_images/base_04_var_33.jpg", "../../images/processed_images/base_04_var_34.jpg",
        "../../images/processed_images/base_04_var_35.jpg", "../../images/processed_images/base_04_var_36.jpg",
        "../../images/processed_images/base_04_var_37.jpg", "../../images/processed_images/base_04_var_38.jpg",
        "../../images/processed_images/base_04_var_39.jpg", "../../images/processed_images/base_04_var_40.jpg",
        "../../images/processed_images/base_07_var_61.jpg", "../../images/processed_images/base_07_var_62.jpg",
        "../../images/processed_images/base_07_var_63.jpg", "../../images/processed_images/base_07_var_64.jpg",
        "../../images/processed_images/base_07_var_65.jpg", "../../images/processed_images/base_07_var_66.jpg",
        "../../images/processed_images/base_07_var_67.jpg", "../../images/processed_images/base_07_var_68.jpg",
        "../../images/processed_images/base_07_var_69.jpg", "../../images/processed_images/base_07_var_70.jpg"
    ],
    "happy": [
        "../../images/processed_images/base_01_var_01.jpg", "../../images/processed_images/base_01_var_02.jpg",
        "../../images/processed_images/base_01_var_03.jpg", "../../images/processed_images/base_01_var_04.jpg",
        "../../images/processed_images/base_01_var_05.jpg", "../../images/processed_images/base_01_var_06.jpg",
        "../../images/processed_images/base_01_var_07.jpg", "../../images/processed_images/base_01_var_08.jpg",
        "../../images/processed_images/base_01_var_09.jpg", "../../images/processed_images/base_01_var_10.jpg",
        "../../images/processed_images/base_03_var_21.jpg", "../../images/processed_images/base_03_var_22.jpg",
        "../../images/processed_images/base_03_var_23.jpg", "../../images/processed_images/base_03_var_24.jpg",
        "../../images/processed_images/base_03_var_25.jpg", "../../images/processed_images/base_03_var_26.jpg",
        "../../images/processed_images/base_03_var_27.jpg", "../../images/processed_images/base_03_var_28.jpg",
        "../../images/processed_images/base_03_var_29.jpg", "../../images/processed_images/base_03_var_30.jpg",
        "../../images/processed_images/base_05_var_41.jpg", "../../images/processed_images/base_05_var_42.jpg",
        "../../images/processed_images/base_05_var_43.jpg", "../../images/processed_images/base_05_var_44.jpg",
        "../../images/processed_images/base_05_var_45.jpg", "../../images/processed_images/base_05_var_46.jpg",
        "../../images/processed_images/base_05_var_47.jpg", "../../images/processed_images/base_05_var_48.jpg",
        "../../images/processed_images/base_05_var_49.jpg", "../../images/processed_images/base_05_var_50.jpg",
        "../../images/processed_images/base_08_var_71.jpg", "../../images/processed_images/base_08_var_72.jpg",
        "../../images/processed_images/base_08_var_73.jpg", "../../images/processed_images/base_08_var_74.jpg",
        "../../images/processed_images/base_08_var_75.jpg", "../../images/processed_images/base_08_var_76.jpg",
        "../../images/processed_images/base_08_var_77.jpg", "../../images/processed_images/base_08_var_78.jpg",
        "../../images/processed_images/base_08_var_79.jpg", "../../images/processed_images/base_08_var_80.jpg"
    ],
    "sad": [
        "../../images/processed_images/base_02_var_11.jpg", "../../images/processed_images/base_02_var_12.jpg",
        "../../images/processed_images/base_02_var_13.jpg", "../../images/processed_images/base_02_var_14.jpg",
        "../../images/processed_images/base_02_var_15.jpg", "../../images/processed_images/base_02_var_16.jpg",
        "../../images/processed_images/base_02_var_17.jpg", "../../images/processed_images/base_02_var_18.jpg",
        "../../images/processed_images/base_02_var_19.jpg", "../../images/processed_images/base_02_var_20.jpg",
        "../../images/processed_images/base_06_var_51.jpg", "../../images/processed_images/base_06_var_52.jpg",
        "../../images/processed_images/base_06_var_53.jpg", "../../images/processed_images/base_06_var_54.jpg",
        "../../images/processed_images/base_06_var_55.jpg", "../../images/processed_images/base_06_var_56.jpg",
        "../../images/processed_images/base_06_var_57.jpg", "../../images/processed_images/base_06_var_58.jpg",
        "../../images/processed_images/base_06_var_59.jpg", "../../images/processed_images/base_06_var_60.jpg",
        "../../images/processed_images/base_06_var_51.jpg", "../../images/processed_images/base_06_var_52.jpg",
        "../../images/processed_images/base_06_var_53.jpg", "../../images/processed_images/base_06_var_54.jpg",
        "../../images/processed_images/base_06_var_55.jpg", "../../images/processed_images/base_06_var_56.jpg",
        "../../images/processed_images/base_06_var_57.jpg", "../../images/processed_images/base_06_var_58.jpg",
        "../../images/processed_images/base_06_var_59.jpg", "../../images/processed_images/base_06_var_60.jpg",
        "../../images/processed_images/base_04_var_31.jpg", "../../images/processed_images/base_04_var_32.jpg",
        "../../images/processed_images/base_04_var_33.jpg", "../../images/processed_images/base_04_var_34.jpg",
        "../../images/processed_images/base_04_var_35.jpg", "../../images/processed_images/base_04_var_36.jpg",
        "../../images/processed_images/base_04_var_37.jpg", "../../images/processed_images/base_04_var_38.jpg",
        "../../images/processed_images/base_04_var_39.jpg", "../../images/processed_images/base_04_var_40.jpg"
    ],
    "angry": [
        "../../images/processed_images/base_01_var_01.jpg", "../../images/processed_images/base_01_var_02.jpg",
        "../../images/processed_images/base_01_var_03.jpg", "../../images/processed_images/base_01_var_04.jpg",
        "../../images/processed_images/base_01_var_05.jpg", "../../images/processed_images/base_01_var_06.jpg",
        "../../images/processed_images/base_01_var_07.jpg", "../../images/processed_images/base_01_var_08.jpg",
        "../../images/processed_images/base_01_var_09.jpg", "../../images/processed_images/base_01_var_10.jpg",
        "../../images/processed_images/base_02_var_11.jpg", "../../images/processed_images/base_02_var_12.jpg",
        "../../images/processed_images/base_02_var_13.jpg", "../../images/processed_images/base_02_var_14.jpg",
        "../../images/processed_images/base_02_var_15.jpg", "../../images/processed_images/base_02_var_16.jpg",
        "../../images/processed_images/base_02_var_17.jpg", "../../images/processed_images/base_02_var_18.jpg",
        "../../images/processed_images/base_02_var_19.jpg", "../../images/processed_images/base_02_var_20.jpg",
        "../../images/processed_images/base_04_var_31.jpg", "../../images/processed_images/base_04_var_32.jpg",
        "../../images/processed_images/base_04_var_33.jpg", "../../images/processed_images/base_04_var_34.jpg",
        "../../images/processed_images/base_04_var_35.jpg", "../../images/processed_images/base_04_var_36.jpg",
        "../../images/processed_images/base_04_var_37.jpg", "../../images/processed_images/base_04_var_38.jpg",
        "../../images/processed_images/base_04_var_39.jpg", "../../images/processed_images/base_04_var_40.jpg",
        "../../images/processed_images/base_01_var_01.jpg", "../../images/processed_images/base_01_var_02.jpg",
        "../../images/processed_images/base_01_var_03.jpg", "../../images/processed_images/base_01_var_04.jpg",
        "../../images/processed_images/base_01_var_05.jpg", "../../images/processed_images/base_01_var_06.jpg",
        "../../images/processed_images/base_01_var_07.jpg", "../../images/processed_images/base_01_var_08.jpg",
        "../../images/processed_images/base_01_var_09.jpg", "../../images/processed_images/base_01_var_10.jpg"
    ],
    "disgust": [
        "../../images/processed_images/base_02_var_11.jpg", "../../images/processed_images/base_02_var_12.jpg",
        "../../images/processed_images/base_02_var_13.jpg", "../../images/processed_images/base_02_var_14.jpg",
        "../../images/processed_images/base_02_var_15.jpg", "../../images/processed_images/base_02_var_16.jpg",
        "../../images/processed_images/base_02_var_17.jpg", "../../images/processed_images/base_02_var_18.jpg",
        "../../images/processed_images/base_02_var_19.jpg", "../../images/processed_images/base_02_var_20.jpg",
        "../../images/processed_images/base_06_var_51.jpg", "../../images/processed_images/base_06_var_52.jpg",
        "../../images/processed_images/base_06_var_53.jpg", "../../images/processed_images/base_06_var_54.jpg",
        "../../images/processed_images/base_06_var_55.jpg", "../../images/processed_images/base_06_var_56.jpg",
        "../../images/processed_images/base_06_var_57.jpg", "../../images/processed_images/base_06_var_58.jpg",
        "../../images/processed_images/base_06_var_59.jpg", "../../images/processed_images/base_06_var_60.jpg",
        "../../images/processed_images/base_06_var_51.jpg", "../../images/processed_images/base_06_var_52.jpg",
        "../../images/processed_images/base_06_var_53.jpg", "../../images/processed_images/base_06_var_54.jpg",
        "../../images/processed_images/base_06_var_55.jpg", "../../images/processed_images/base_06_var_56.jpg",
        "../../images/processed_images/base_06_var_57.jpg", "../../images/processed_images/base_06_var_58.jpg",
        "../../images/processed_images/base_06_var_59.jpg", "../../images/processed_images/base_06_var_60.jpg",
        "../../images/processed_images/base_02_var_11.jpg", "../../images/processed_images/base_02_var_12.jpg",
        "../../images/processed_images/base_02_var_13.jpg", "../../images/processed_images/base_02_var_14.jpg",
        "../../images/processed_images/base_02_var_15.jpg", "../../images/processed_images/base_02_var_16.jpg",
        "../../images/processed_images/base_02_var_17.jpg", "../../images/processed_images/base_02_var_18.jpg",
        "../../images/processed_images/base_02_var_19.jpg", "../../images/processed_images/base_02_var_20.jpg"
    ],
    "fear": [
        "../../images/processed_images/base_06_var_51.jpg", "../../images/processed_images/base_06_var_52.jpg",
        "../../images/processed_images/base_06_var_53.jpg", "../../images/processed_images/base_06_var_54.jpg",
        "../../images/processed_images/base_06_var_55.jpg", "../../images/processed_images/base_06_var_56.jpg",
        "../../images/processed_images/base_06_var_57.jpg", "../../images/processed_images/base_06_var_58.jpg",
        "../../images/processed_images/base_06_var_59.jpg", "../../images/processed_images/base_06_var_60.jpg",
        "../../images/processed_images/base_07_var_61.jpg", "../../images/processed_images/base_07_var_62.jpg",
        "../../images/processed_images/base_07_var_63.jpg", "../../images/processed_images/base_07_var_64.jpg",
        "../../images/processed_images/base_07_var_65.jpg", "../../images/processed_images/base_07_var_66.jpg",
        "../../images/processed_images/base_07_var_67.jpg", "../../images/processed_images/base_07_var_68.jpg",
        "../../images/processed_images/base_07_var_69.jpg", "../../images/processed_images/base_07_var_70.jpg",
        "../../images/processed_images/base_01_var_01.jpg", "../../images/processed_images/base_01_var_02.jpg",
        "../../images/processed_images/base_01_var_03.jpg", "../../images/processed_images/base_01_var_04.jpg",
        "../../images/processed_images/base_01_var_05.jpg", "../../images/processed_images/base_01_var_06.jpg",
        "../../images/processed_images/base_01_var_07.jpg", "../../images/processed_images/base_01_var_08.jpg",
        "../../images/processed_images/base_01_var_09.jpg", "../../images/processed_images/base_01_var_10.jpg",
        "../../images/processed_images/base_02_var_11.jpg", "../../images/processed_images/base_02_var_12.jpg",
        "../../images/processed_images/base_02_var_13.jpg", "../../images/processed_images/base_02_var_14.jpg",
        "../../images/processed_images/base_02_var_15.jpg", "../../images/processed_images/base_02_var_16.jpg",
        "../../images/processed_images/base_02_var_17.jpg", "../../images/processed_images/base_02_var_18.jpg",
        "../../images/processed_images/base_02_var_19.jpg", "../../images/processed_images/base_02_var_20.jpg"
    ],
    "surprise": [
        "../../images/processed_images/base_05_var_41.jpg", "../../images/processed_images/base_05_var_42.jpg",
        "../../images/processed_images/base_05_var_43.jpg", "../../images/processed_images/base_05_var_44.jpg",
        "../../images/processed_images/base_05_var_45.jpg", "../../images/processed_images/base_05_var_46.jpg",
        "../../images/processed_images/base_05_var_47.jpg", "../../images/processed_images/base_05_var_48.jpg",
        "../../images/processed_images/base_05_var_49.jpg", "../../images/processed_images/base_05_var_50.jpg",
        "../../images/processed_images/base_07_var_61.jpg", "../../images/processed_images/base_07_var_62.jpg",
        "../../images/processed_images/base_07_var_63.jpg", "../../images/processed_images/base_07_var_64.jpg",
        "../../images/processed_images/base_07_var_65.jpg", "../../images/processed_images/base_07_var_66.jpg",
        "../../images/processed_images/base_07_var_67.jpg", "../../images/processed_images/base_07_var_68.jpg",
        "../../images/processed_images/base_07_var_69.jpg", "../../images/processed_images/base_07_var_70.jpg",
        "../../images/processed_images/base_08_var_71.jpg", "../../images/processed_images/base_08_var_72.jpg",
        "../../images/processed_images/base_08_var_73.jpg", "../../images/processed_images/base_08_var_74.jpg",
        "../../images/processed_images/base_08_var_75.jpg", "../../images/processed_images/base_08_var_76.jpg",
        "../../images/processed_images/base_08_var_77.jpg", "../../images/processed_images/base_08_var_78.jpg",
        "../../images/processed_images/base_08_var_79.jpg", "../../images/processed_images/base_08_var_80.jpg",
        "../../images/processed_images/base_05_var_41.jpg", "../../images/processed_images/base_05_var_42.jpg",
        "../../images/processed_images/base_05_var_43.jpg", "../../images/processed_images/base_05_var_44.jpg",
        "../../images/processed_images/base_05_var_45.jpg", "../../images/processed_images/base_05_var_46.jpg",
        "../../images/processed_images/base_05_var_47.jpg", "../../images/processed_images/base_05_var_48.jpg",
        "../../images/processed_images/base_05_var_49.jpg", "../../images/processed_images/base_05_var_50.jpg"
    ]
}

# Variables para controlar la frecuencia de la animación
current_emotion = "neutral"
animation_thread = None
animation_running = False

def animate_avatar():
    """Mantiene la ventana de avatar abierta y actualiza las imágenes según la emoción detectada."""
    global current_emotion, animation_running
    cv2.namedWindow("Avatar Animation", cv2.WINDOW_NORMAL)
    while animation_running:
        images = IMAGE_PATHS.get(current_emotion, IMAGE_PATHS["neutral"])
        for img_path in images:
            if not animation_running:
                break
            image = cv2.imread(img_path)
            if image is not None:
                cv2.imshow("Avatar Animation", image)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    animation_running = False
                    break
    cv2.destroyWindow("Avatar Animation")

def start_animation_thread():
    """Inicia el hilo de animación si no está en ejecución."""
    global animation_thread, animation_running
    if animation_thread is None or not animation_thread.is_alive():
        animation_running = True
        animation_thread = threading.Thread(target=animate_avatar)
        animation_thread.start()

def stop_animation_thread():
    """Detiene el hilo de animación."""
    global animation_running
    animation_running = False
    if animation_thread and animation_thread.is_alive():
        animation_thread.join()

def start_real_time_emotion_detection():
    global current_emotion
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    start_animation_thread()  # Inicia la animación al comienzo

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta emociones en el fotograma actual sin mostrar la ventana de FER
        emotion, score = detector.top_emotion(frame)

        # Actualiza la emoción actual solo si es diferente
        if emotion and emotion != current_emotion:
            print(f"Emotion detected: {emotion}, Confidence: {score * 100:.2f}%")
            current_emotion = emotion  # Cambia la emoción actual para la animación

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_animation_thread()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_real_time_emotion_detection()