from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
def euclidean(a, b):
    return np.linalg.norm(a-b)
def EAR(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


#########################
offsetPercentageW = 20
offsetPercentageH = 20
confidence = 0.8
camWidth = 640
camHeight = 480
threshhold = 0.25
#########################
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceMeshDetector(maxFaces=1)
while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        leftEye = [face[i] for i in [33, 160, 158, 133, 153, 144]]
        rightEye = [face[i] for i in [362, 385, 387, 263, 373, 380]]
        ear_left = EAR(leftEye)
        ear_right = EAR(rightEye)
        if ear_left < threshhold or ear_right < threshhold:
            print("Blink Detected")
        for (x, y) in leftEye + rightEye:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Face Mesh", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
