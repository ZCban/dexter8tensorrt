import time
import numpy as np
import torch
from utils.utils_trt import TRTModule
import kmNet
from utils.utils import check_cuda_device, draw_visuals, count_fps,det_postprocess,box_convert_numpy
import bettercam
from rich import print
import sys
import cv2
import win32api
import os


screenshot = 512
center = screenshot/2
model_file = 'yolov8n.engine'
min_conf = 0.49
fps_limit=90
movespeed=1.8
visual = True
fpscount=False

current_target = None


check_cuda_device()

kmNet.init('192.168.2.188','1408','9FC05414')

# Ottiene il percorso della directory in cui si trova lo script corrente
current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'models')

model_path = os.path.join(models_path, model_file)
print(model_path)


left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGRA")
cam.start(region=region,video_mode=True,target_fps=fps_limit)

# Load model
model = TRTModule(model_path, device=0)

# Inizializza una variabile di conteggio fps
fps_counter = 0
start_time = time.time()

while True:   
    img = cam.get_latest_frame()  # Cattura lo screenshot


    tensor = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0
    tensor = torch.as_tensor(tensor, device='cuda')
    data = model(tensor)
    bboxes = det_postprocess(data, confidence_threshold=min_conf)

    bboxes = box_convert_numpy(bboxes, 'xyxy', 'cxcywh')
    targets_data = bboxes[:, :4]
    
    if len(targets_data) > 0 : 
        dist_from_center = np.sqrt(np.sum((targets_data[:, :1] - center)**2, axis=1))
        min_dist_idx = np.argmin(dist_from_center)
        current_target = targets_data[min_dist_idx]
        delta_x = current_target[0] - center
        delta_y = current_target[1] - center
        #delta_y -= (current_target[3]/4)+((current_target[3]/3)/3)
        delta_y -= (current_target[3]/3)
        if  win32api.GetKeyState(0x05)<0:
            kmNet.move(int(delta_x/movespeed),int(delta_y/movespeed))
            if -2.4 <= delta_x/movespeed <=2.4 and -1 <= delta_y/movespeed <= 1:
                kmNet.left(1)
                kmNet.left(0)         
            
    if visual:
        draw_visuals(img, bboxes, current_target)

    if fpscount:
        start_time, fps_counter = count_fps(start_time, fps_counter)

    

