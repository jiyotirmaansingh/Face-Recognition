import cv2
import cv2.data

modelPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
trainedMachine = cv2.CascadeClassifier(modelPath)
camera = cv2.VideoCapture(0)

while True:
    status, frame = camera.read()
    faces = trainedMachine.detectMultiScale(frame, 1.3, 5)
    for face in faces: 
        x, y, w, h = face
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("image", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()


//python face_detection.py
