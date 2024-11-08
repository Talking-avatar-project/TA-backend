import cv2
from fer import FER

def start_real_time_emotion_detection():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta emociones en el fotograma actual
        emotion, score = detector.top_emotion(frame)
        label = f"{emotion}: {score * 100:.2f}%" if emotion else "No face detected"

        # Muestra el resultado en el fotograma
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Emotion Detection", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_real_time_emotion_detection()
