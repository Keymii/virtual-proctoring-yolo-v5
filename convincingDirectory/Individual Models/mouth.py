import dlib
import cv2 as cv
import numpy as np
from math import sqrt
import time

vid = cv.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()
face_landmarks = dlib.shape_predictor('shape_68.dat')

time.sleep(0.5)

while(True):
    _, frame = vid.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray_frame)
    for face in faces:
        landmarks = face_landmarks(gray_frame, face)

        # Upper lip
        x1 = landmarks.part(51).x
        y1 = landmarks.part(51).y
        
        x2 = landmarks.part(62).x
        y2 = landmarks.part(62).y

        x = x2 - x1
        y = y2 - y1

        upper_lip_thickness = sqrt(x**2 + y**2)

        # Lower lip
        a1 = landmarks.part(66).x
        b1 = landmarks.part(66).y
        
        a2 = landmarks.part(57).x
        b2 = landmarks.part(57).y

        a = a2 - a1
        b = b2 - b1

        lower_lip_thickness = sqrt(a**2 + b**2)

        thickness = max(upper_lip_thickness, lower_lip_thickness)

        # cv.circle(frame, (x1, y1), 2, (255, 255, 255), 2)
        # cv.circle(frame, (x2, y2), 2, (255, 255, 255), 2)
        # cv.circle(frame, (a1, b1), 2, (255, 255, 255), 2)
        # cv.circle(frame, (a2, b2), 2, (255, 255, 255), 2)

        if sqrt((a1 - x2)**2 + (b1 - y2)**2) > thickness:
            cv.putText(frame, "Please close your mouth!", (100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))



    cv.imshow("FACE LANDMARKS", frame)

    if cv.waitKey(1) == 97:
        break

vid.release()
cv.destroyAllWindows()
