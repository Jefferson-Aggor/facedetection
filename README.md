# Python RealTime Face Detection Using Haar Cascade 

Haar Cascade is a machine learning-based object detection algorithm used in OpenCV (CV2) for detecting objects like faces, eyes, and cars in images or video streams. It uses a set of pre-trained classifiers to identify features of objects and then detects those objects by analyzing the patterns of light and dark regions in the image. [Read More]('https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html' "Haar Cascade") 

## STEPS
1. Install OPENCV using `pip`
```bash
pip install opencv-python
```

2. Import the openCV library
```py
import cv2
```

3. Import the Haar Cascade Classifier
```py
# We will use the pre-trained Haar Cascade classifier built into OpenCV
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
```

4. Open the webcam for realtime face detection
```py
video_capture = cv2.VideoCapture(0)
# The argument is 0, means we are selecting the default camera on the device.
```

5. Identify the faces in the video Strem and Draw Bounding Boxes around them.
```py
def detect_boundingBox(video):
    gray_scale_img = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray_scale_img, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in face:
        cv2.rectangle(video, (x,y), (x+w, y+h), (0, 0, 255), 4)
    return face
```

6. Create a Loop for Real-Time Face Detection
```py
while True:
    result, video_frame = video_capture.read()

    if result is False:
        break 

    face = detect_boundingBox(video_frame)
    cv2.imshow("Face Detection", video_frame)

    # Quit the application when the key 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

7. Release and Destroy All Windows
```
video_capture.release()
cv2.destroyAllWindows()
```

