import cv2
from fer import FER  # Facial Expression Recognition

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load face cascade
face_cascade = cv2.CascadeClassifier('multimodal/haarcascade_frontalface_default.xml')

# Initialize FER emotion detector
emotion_detector = FER(mtcnn=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Process each face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop face region for emotion detection
        face_crop = frame[y:y + h, x:x + w]

        # Detect emotion
        emotion_result = emotion_detector.detect_emotions(face_crop)
        if emotion_result:
            top_emotion, score = emotion_detector.top_emotion(face_crop)
            cv2.putText(frame, f'{top_emotion} ({score:.2f})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display output
    cv2.imshow('Multimodal Face Detection (Faces + Emotions)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
