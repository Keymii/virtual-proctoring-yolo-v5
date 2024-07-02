import yolov5
import cv2 as cv

model = yolov5.load('yolov5n.pt')
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

vid = cv.VideoCapture(0)

while(True):
    _, frame = vid.read()
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred_frame = cv.GaussianBlur(grey_frame, (3, 3), 0)

    results = model(blurred_frame)

    # results.show()
    # print(results)
    # predictions = results.pred[0]
    # print(predictions)

    string_results = str(results)
    # print(string_results)
    if string_results.find("cell phone") != -1:
        cv.putText(frame, "Cell phone detected!", (100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

    cv.imshow("Video", frame)

    if cv.waitKey(1) == 27:
        break

vid.release()
cv.destroyAllWindows()