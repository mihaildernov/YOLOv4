import cv2 as cv
import time
import numpy as np
from collections import Counter

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('coco.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('yolov4.weights', 'yolov4.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv.VideoCapture(0)
starting_time = time.time()
frame_counter = 0

while True:
    ret, frame = cap.read()
    frame_counter += 1

    if ret == False:
        break

    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime

    cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        cv.rectangle(frame, box, color, 1)
        scor = ["%.2f" % score]
        label = class_name[classid], " ", scor[0]
        lab = ''.join(label)
        cv.putText(frame, lab, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv.imshow('frame', frame)
        c = Counter(classes)
        print(f"Обнаружены объекты: {c}")
        print()

    centres = []
    x1 = 320
    y1 = 240

    for i in boxes:
        res = i.astype(np.uint8)
        moment = cv.moments(res)
        x2 = int(moment['m10'] / moment['m00']);
        y2 = int(moment['m01'] / moment['m00'])
        centres.append((x2, y2))
        x = x2 - x1;
        y = y2 - y1
        print(f"Координаты (x: {x}, y: {y})")
        print()

    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
