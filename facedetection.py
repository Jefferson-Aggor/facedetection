import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface.xml"
)

video_capture = cv2.VideoCapture(0)

def detect_boundingBox(video):
    gray_scale_img = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray_scale_img, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in face:
        cv2.rectangle(video, (x,y), (x+w, y+h), (0, 0, 255), 4)
    return face

while True:
    result, video_frame = video_capture.read()

    if result is False:
        break 

    face = detect_boundingBox(video_frame)
    cv2.imshow("Face Detection", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()