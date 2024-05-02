import time
import numpy as np
import torch
from utils.utils_trt import TRTModule
import kmNet
from utils.utils import check_cuda_device, draw_visuals, det_postprocess6, box_convert_numpy,count_fps
import bettercam
from rich import print
import sys
import cv2
import win32api
import os

# Dimensions of the full screenshot
screenshot = 448
center = screenshot / 2
center_square_size = 380
half_square = center_square_size / 2

# Configuration parameters
model_file = '448.engine'
min_conf = 0.50
fps_limit = 60
movespeed = 1.9
visual = False
fpscount = True

# Initialize network and CUDA device
check_cuda_device()
kmNet.init('192.168.2.188', '1408', '9FC05414')

# Calculate the path of the current directory and model file
current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'models')
model_path = os.path.join(models_path, model_file)
print(model_path)

left, top = (1920 - screenshot) // 2, (1080 - screenshot) // 2
right, bottom = left + screenshot, top + screenshot
region = (left, top, right, bottom)
cam=bettercam.create(output_idx=0, output_color="BGRA")
cam.start(region=region,video_mode=True,target_fps=fps_limit)

# Load the model
model = TRTModule(model_path, device=0)
current_target=None

# Inizializza una variabile di conteggio fps
fps_counter = 0
start_time = time.time()

while True:
    img = cam.get_latest_frame()  # Capture the screenshot

    # Preprocess the image to tensor
    tensor = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0
    tensor = torch.as_tensor(tensor, device='cuda')
    
    # Model inference
    data = model(tensor)
    bboxes = det_postprocess6(data, confidence_threshold=min_conf)
    targets_data = bboxes[:, :4]
    
    if len(targets_data) > 0 and win32api.GetKeyState(0x05) < 0:
        current_target = targets_data[0]
        delta_x = current_target[0] - center
        delta_y = (current_target[1] - center)-(current_target[3]/4)
        #delta_y -= (current_target[3]/3)
        
        # Check if the target is within the central square
        if -half_square <= delta_x <= half_square and -half_square <= delta_y <= half_square :
            kmNet.move(int(delta_x/movespeed), int(delta_y/movespeed))
            if -3 <= delta_x/movespeed <= 3 and -2 <= delta_y/movespeed <= 2:
                kmNet.left(1)
                kmNet.left(0)
            if not -3 <= delta_x/movespeed <= 3 and -2 <= delta_y/movespeed <= 2:
                kmNet.move(int(delta_x/movespeed), int(delta_y/movespeed))

    # Visualize the output if enabled
    if visual:
        draw_visuals(img, bboxes, current_target)

    if fpscount:
        start_time, fps_counter = count_fps(start_time, fps_counter)
