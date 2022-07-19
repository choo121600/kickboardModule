import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import threading
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots2 import plot_one_box
from utils.torch_utils2 import select_device, time_synchronized
from imutils.video import WebcamVideoStream
import imutils

import time # time 라이브러리
SOURCE = 'fire.jpg'
WEIGHTS = 'yolov5s.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

'''for i in range(1280):
    for j in range(480):
        result[i][j]=0'''
result=np.zeros((960,1280,3),dtype=np.uint8)
print(result.shape)


source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()  # to FP16
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


outputFrame = None
lock = threading.Lock()
#vs = VideoStream('rtsp://admin:infoadmin01@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0').start()
frame0 = WebcamVideoStream(src='rtsp://admin:infoadmin01@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0').start()
#frame1 = WebcamVideoStream(src=2).start()

time.sleep(2.0)




def detect():
   # frame0 = WebcamVideoStream(src=0).start()
   # frame1 = WebcamVideoStream(src=1).start()
    prevTime = 0  # 이전 시간을 저장할 변수
    while 1:
        curTime = time.time()
        img0 = frame0.read()




        #img0 = imutils.resize(img0, width=400)
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
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        str = "FPS : %0.1f" % fps
        cv2.putText(img0, str, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow(source, img0)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    frame0.stop()
    cv2.destroyAllWindows()
def thread_run():
    threading.Timer(0.001, detect()).start()


if __name__ == '__main__':
    #detect()

    '''#detectInit()
    t = threading.Thread(target=detect)
    t.daemon = True
    t.start()

    frame0.stop()
    cv2.destroyAllWindows()
    #with torch.no_grad():detect()
    #check_requirements(exclude=('pycocotools', 'thop'))
    #with torch.no_grad():detect(img0)
    #detect(img0)'''

    t = threading.Thread(target=detect)
    t.start()

    #frame0.stop()
    #thread_run()
