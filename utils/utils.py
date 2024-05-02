import time
import torch
import json
import ctypes
from rich import print
from ctypes.wintypes import LARGE_INTEGER
import sys
import cv2
import os
import pickle
import warnings
from typing import List, OrderedDict, Tuple, Union
import torch
import numpy as np
import tensorrt as trt
from collections import namedtuple
from typing import List, Optional, Tuple, Union
from pathlib import Path

def accurate_timing(duration_ms: int) -> float:
    kernel32 = ctypes.windll.kernel32

    INFINITE = 0xFFFFFFFF
    WAIT_FAILED = 0xFFFFFFFF
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002

    # Call WaitableTimer w/ CREATE_WAITABLE_TIMER_HIGH_RESOLUTION Flag
    handle = kernel32.CreateWaitableTimerExW(None, None, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, 0x1F0003)
    res = kernel32.SetWaitableTimer(handle, ctypes.byref(LARGE_INTEGER(int(duration_ms * -10000))), 0, None, None, 0,)
    
    start_time = time.perf_counter()
    res = kernel32.WaitForSingleObject(handle, INFINITE)
    kernel32.CancelWaitableTimer(handle)

    return (time.perf_counter() - start_time) * 1000



def check_cuda_device():
    # Verifica la disponibilità di CUDA
    if torch.cuda.is_available():
        cuda_status = "[yellow]CUDA device found:[/yellow] [orange]{}[/orange]".format(torch.cuda.get_device_name(0))
        print(cuda_status)
    else:
        cuda_status = "[red]No CUDA device found.[/red]"
        print(cuda_status)

def draw_visuals(img, bboxes, current_target):
    # Draw bounding box around the detected object
    if len(bboxes) > 0:
        cx, cy, w, h = current_target
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with rectangles
    cv2.imshow("Object Detection", img)
    cv2.waitKey(1)



def count_fps(start_time, fps_counter):
    elapsed_time = time.time() - start_time
    fps_counter += 1

    if elapsed_time >= 1.0:
        fps = fps_counter / elapsed_time
        sys.stdout.write(f"\rFPS: {fps:.2f}")
        sys.stdout.flush()
        fps_counter = 0
        start_time = time.time()

    return start_time, fps_counter

def det_postprocess6(data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                     confidence_threshold: float = 0.42, 
                     class_id: int = 0,  
                     max_results: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    
    # Applica i filtri di soglia di confidenza e ID della classe
    selected = (scores >= confidence_threshold) & (labels == class_id)
    
    # Applica la selezione
    scores_selected = scores[selected]
    #labels_selected = labels[selected]
    bboxes_selected = bboxes[selected]
    
    # Utilizza torch.topk per trovare i punteggi più alti e i loro indici
    #top_scores, top_indices = scores_selected.topk(min(max_results, scores_selected.size(0)), largest=True)
    #top_bboxes = bboxes_selected[top_indices]
    #top_labels = labels_selected[top_indices]
    #return bboxes_selected
    # Avoid moving to CPU prematurely; keep computations on GPU
    x1, y1, x2, y2 = bboxes_selected[:, 0], bboxes_selected[:, 1], bboxes_selected[:, 2], bboxes_selected[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    new_boxes = torch.stack([cx, cy, w, h], dim=1)
        
    return new_boxes.cpu().numpy()
    




def box_convert_cupy(boxes, from_mode, to_mode):
    if from_mode == to_mode:
        return boxes

    if from_mode == 'xyxy' and to_mode == 'cxcywh':
        x1, y1, x2, y2 = cp.split(boxes, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return cp.concatenate([cx, cy, w, h], axis=-1)

    # Add other conversion modes as needed

    raise ValueError(f"Conversion from {from_mode} to {to_mode} not supported")


def box_convert_numpy(boxes, from_mode, to_mode):
    # Avoid moving to CPU prematurely; keep computations on GPU
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    new_boxes = torch.stack([cx, cy, w, h], dim=1)
        
    return new_boxes.cpu().numpy() 

def box_convert_torch(boxes, from_mode, to_mode):
    if from_mode == to_mode:
        return boxes

    if from_mode == 'xyxy' and to_mode == 'cxcywh':
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.cat([cx, cy, w, h], dim=-1)

    # Add other conversion modes as needed

    raise ValueError(f"Conversion from {from_mode} to {to_mode} not supported")




def box_convert(boxes, from_mode, to_mode):
    if from_mode == to_mode:
        return boxes

    if from_mode == 'xyxy' and to_mode == 'cxcywh':
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.concatenate([cx, cy, w, h], axis=-1)

    # Add other conversion modes as needed

    raise ValueError(f"Conversion from {from_mode} to {to_mode} not supported")
