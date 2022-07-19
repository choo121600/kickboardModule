import math
from detects.helmet import *
from detects.gps import *
from settings import *

class Check:
    def __init__(self):
        self.helmet = Helmet()
        self.gps = Gps()
        self.move = 'stop'

    def set_speed(self, helmet_state, tim):
        global max_speed
        if helmet_state == 'on' or (BOHO[0][0] < self.gps.lat < BOHO[1][0] and BOHO[0][1] < self.gps.lon < BOHO[1][1]):
            if max_speed > 19.8:
                max_speed = 20
            else:
                var_speed = 10 / (1 + math.exp(-0.05 * (tim%36) + 5))
                max_speed = max_speed + var_speed
        if helmet_state == 'off':
            if max_speed < 10.2:
                max_speed = 10
            else:
                var_speed = 10 / (1 + math.exp(-0.05 * (tim%36) + 5))
                max_speed = max_speed - var_speed
        else:
            self.helmet.helmet_state = 'non'
        print(max_speed)

    def update(self, img0, xyxy, detect_name, draw_color, sec):
        self.helmet.detect_helmet(img0, xyxy, detect_name, draw_color)
        self.gps.detect_gps()
        self.set_speed(self.helmet.helmet_state, sec)