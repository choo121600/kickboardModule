import cv2
import torch
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
import time # time 라이브러리

from check import *
from settings import *


cap= cv2.VideoCapture(0)
prevTime = 0
timer = 0

while cap.isOpened():
    check = Check()
    curTime = time.time()
    ret,img0 = cap.read()
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=AUGMENT)[0]
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
    det = pred[0]
    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string 
        for *xyxy, conf, cls in reversed(det):
            draw_color = colors[int(cls)]
            detect_name = names[int(cls)]
            check.update(img0, xyxy, detect_name, draw_color, sec)

    sec = curTime - prevTime
    prevTime = curTime
    timer += sec
    fps = 1 / (sec)
    str = "FPS : %0.1f" % fps
    cv2.putText(img0, str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow(source, img0)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break