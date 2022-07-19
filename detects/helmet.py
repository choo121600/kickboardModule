from settings import *
from utils.plots2 import plot_one_box

class Helmet:
    def __init__(self):
        self.helmet_state = "off"
    
    def detect_helmet(self, img0, xyxy, detect_name, draw_color):
        if detect_name == 'on':
            label = detect_name
            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
            self.helmet_state = 'on'
        else:
            self.helmet_state = 'off'
