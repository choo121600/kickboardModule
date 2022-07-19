# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots2 import plot_one_box
from utils.torch_utils2 import select_device, time_synchronized
import boto3
import os
import sys
import numpy as np
import datetime
from boto3.dynamodb.conditions import Key,Attr

AWS_ACCESS_KEY="AKIARFAXSUTCI2WOJE6O"
AWS_SECRET_KEY="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"
AWS_REGION_NAME='ap-northeast-2'
TABLE_NAME="smart_event"
PARTITION_KEY="date"
SORT_KEY="id"

client=boto3.resource("dynamodb",
                     aws_access_key_id=AWS_ACCESS_KEY,
                     aws_secret_access_key=AWS_SECRET_KEY,
                     region_name=AWS_REGION_NAME)
table=client.Table(TABLE_NAME)

TABLE_NAME="smart_cmd"


client=boto3.resource("dynamodb",
                     aws_access_key_id=AWS_ACCESS_KEY,
                     aws_secret_access_key=AWS_SECRET_KEY,
                     region_name=AWS_REGION_NAME)
table2=client.Table(TABLE_NAME)




#old_dt=old_dt



SOURCE=''
WEIGHTS = 'yolov5s.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False



outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
vs = VideoStream('rtsp://admin:infoadmin01@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0').start()
# 'rtsp://admin:infoadmin01@192.168.0.2:554/cam/realmonitor?channel=1&subtype=0'
#src=0
time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")


def detect(frameCount):
    global vs, outputFrame, lock

    source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Load image
    img0 = cv2.imread(source)  # BGR
    assert img0 is not None, 'Image Not Found ' + source
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0]
    print('pred shape:', pred.shape)
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
    det = pred[0]
    print('det shape:', det.shape)
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
        print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')
    print(s)
    cv2.imshow(source, img0)
    cv2.waitKey(0)  # 1 millisecond


def detect_motion(frameCount):
    global vs, outputFrame, lock
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        if total > frameCount:
            motion = md.detect(gray)
            if motion is not None:
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)
        md.update(gray)
        total += 1
        with lock:
            outputFrame = frame.copy()
def detect_motion2(frameCount):
    global vs, outputFrame, lock
    source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    #md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    old_dt = datetime.datetime.now().minute
    while True:

        dt = datetime.datetime.now()
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        img = letterbox(frame, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t0 = time_synchronized()
        pred = model(img, augment=AUGMENT)[0]
        #print('pred shape:', pred.shape)
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
        det = pred[0]
        #print('det shape:', det.shape)
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        Query = Key("id").eq("1")
        response = table2.query(KeyConditionExpression=Query)
        x = response["Items"][0]['command']
        count=0
        if len(det):

            print(len(det))
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):

                label = f'{names[int(cls)]} {conf:.2f}'
                #print(names[int(cls)])
                if int(cls) == int(x):
                    count = count + 1
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
                    
            if old_dt!=dt.minute and count!=0:

                mydt = "{}-{}-{} {}:{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                message = "cmd:{},event:{}".format(int(x), count)
                JSON_RECORD = {
                    PARTITION_KEY: mydt,
                    SORT_KEY: "1",
                    "message": message
                }
                response = table.put_item(Item=JSON_RECORD)
                old_dt = dt.minute


            #print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

        # Stream results
       # print(s)
        #cv2.imshow(source, frame)
        #cv2.waitKey(0)  # 1 millisecond

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        #if total > frameCount:
            #motion = md.detect(gray)
            #if motion is not None:
            #    (thresh, (minX, minY, maxX, maxY)) = motion
             #   cv2.rectangle(frame, (minX, minY), (maxX, maxY),
             #                 (0, 0, 255), 2)
        #md.update(gray)
        total += 1
        with lock:
            outputFrame = frame.copy()
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion2, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
