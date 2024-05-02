import os
import sys
import ctypes
import argparse
from threading import Event
import torch
import tensorrt
from rich import print
import pickle
from collections import OrderedDict
from ultralytics import YOLO
from utils.utils_trt import EngineBuilder

if torch.cuda.is_available():
    print('[green]CUDA device found:', torch.cuda.get_device_name(0))
else:
    print('[red]No CUDA device found.')
    sys.exit(0)

print(f'[green]TensorRT found:[/green] [cyan]{tensorrt.__version__}')

current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'models')
model_path = os.path.join(models_path)
no_engine_check=False
model_inputsize=640

def build_engine(weights: str, input_shape: tuple, output_pkl: str) -> None:
    model = YOLO(weights)
    model.model.fuse()
    YOLOv8 = model.model.model

    strides = YOLOv8[-1].stride.detach().cpu().numpy()
    reg_max = YOLOv8[-1].dfl.conv.weight.shape[1]


    state_dict = OrderedDict(GD=model.model.yaml['depth_multiple'],
                            GW=model.model.yaml['width_multiple'],
                            strides=strides,
                            reg_max=reg_max)


    for name, value in YOLOv8.state_dict().items():
        value = value.detach().cpu().numpy()
        i = int(name.split('.')[0])
        layer = YOLOv8[i]
        module_name = layer.type.split('.')[-1]
        stem = module_name + '.' + name
        state_dict[stem] = value

    with open(output_pkl, 'wb') as f:
        pickle.dump(state_dict, f)

    builder = EngineBuilder(output_pkl, 'cuda:0')
    builder.build(fp16=True, input_shape=input_shape,iou_thres=.5, conf_thres=.5,topk=10)

if model_path.endswith('pt'):
    print('[yellow]Specified YOLO.pt when a TensorRT engine is needed...Loading TensorRT engine!')
    model_path = model_path.replace('pt', 'engine')
        
if not no_engine_check:
    print('[yellow]Checking engines...')
    rel_path = 'models/'
    weights = [n for n in os.listdir(rel_path) if n.endswith('pt')]
    engines = [n for n in os.listdir(rel_path) if n.endswith('engine')]
    for file in weights:
        engine_filename = file.replace('pt', 'engine')
        if engine_filename not in engines:
            inputsz = model_inputsize
            print(f'[red]{engine_filename} is missing.[/red] [magenta]Building engine from YOLO weights. This may take a while...')
            pkl_filename = engine_filename.replace('engine', 'pkl')
            pt_filename =  engine_filename.replace('engine', 'pt')
            input_shape = (1, 3, inputsz, inputsz)
            build_engine(os.path.join(rel_path, pt_filename), input_shape, os.path.join(rel_path, pkl_filename))
            print(f'[yellow]Removing {pkl_filename}')
            os.remove(os.path.join(rel_path, pkl_filename))
