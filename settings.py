import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device

max_speed = 10

port = '/dev/ttyTHS1' # 시리얼 포트
baud = 9600 # 시리얼 보드레이트(통신속도)

BOHO = [[35.512586, 129.298071], [35.512828, 129.298539]]
inBoho = 0

SOURCE = 'yolo Object Detection'
WEIGHTS = 'hel_best.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False
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
