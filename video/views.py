from django.shortcuts import render
from django.http import StreamingHttpResponse,HttpResponse
from django.views.decorators import gzip
import cv2, threading
import yolov5
import cv2
import dlib
import numpy as np
from math import sqrt
import time
from django.template import loader

# notification=""
model = yolov5.load('yolov5n.pt')
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image


def nothing(x):
    pass

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


#Mask for detecting the eyeballs
def eye_on_mask(mask, side,shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

#Highlight the eyeballs
def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

detector = dlib.get_frontal_face_detector()

#Using the facial landmark predictor from dlib
predictor = dlib.shape_predictor('shape_68.dat')
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    thresh = img.copy()

    # cv2.namedWindow('image')
    kernel = np.ones((9, 9), np.uint8)

    # create trackbars for color change
    # cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    # global notification
    while True:
        # notification=""
        img = camera.get_frame()
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray, (3, 3), 0)
        
        #Face Detection
        rects = detector(gray, 1)
        if len(rects) >1:
            # print("ALERT - Multiple faces detected!")
            cv2.putText(img, "ALERT - Multiple faces detected!", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

            # notification+="ALERT - Multiple faces detected!,"
        
        #Model Prediction
        results = model(blurred_frame)
        
        string_results = str(results)
        # print(string_results)
        if string_results.find("cell phone") != -1:
            cv2.putText(img, "Cell phone detected!", (100, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
            # notification+= "Cell phone detected!,"
        #Gaze Detection


        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(mask, left,shape)
            mask = eye_on_mask(mask, right,shape)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            # threshold = cv2.getTrackbarPos('threshold', 'image')
            threshold=66
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2) #1
            thresh = cv2.dilate(thresh, None, iterations=4) #2
            thresh = cv2.medianBlur(thresh, 3) #3
            thresh = cv2.bitwise_not(thresh)
            contouring(thresh[:, 0:mid], mid, img)
            contouring(thresh[:, mid:], mid, img, True)
            # for (x, y) in shape[36:48]:
            #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        # show the image with the face detections + facial landmarks
        # cv2.imshow('eyes', img)
        # cv2.imshow("image", thresh)

        

        #Mouth detection

        for face in rects:
            landmarks = predictor(gray, rect)

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
                cv2.putText(img, "Please close your mouth!", (100, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
                # notification+="Please close your mouth!,"
    
        _,frame=cv2.imencode('.jpg',img)
        frame=frame.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +frame+ b'\r\n')

@gzip.gzip_page
def livefe(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except: 
        print("exception") # This is bad!
        pass

def index(request, *args, **kwargs):
    return render(request, 'index.html')

# def notify(request):
#         global notification
#         template = loader.get_template('index.html')
#         context={
#             'notification':notification,
#         }
#         return HttpResponse(template.render(context, request))