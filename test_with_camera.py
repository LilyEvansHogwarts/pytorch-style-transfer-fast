import torch
import torchvision
import os
import numpy as np
from collections import namedtuple
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model import TransformerNet
from utils import *

def prepare_model(inference_config, device):
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(os.path.join(inference_config['model_binaries_path'], inference_config['model_name']), map_location=torch.device('cpu'))
    state_dict = training_state['state_dict']
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()
    return stylization_model

def prepare_img_for_camera(image):
    img = image.astype(np.float32)
    img /= 255

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        ])
    
    img = transform(img).to(device).unsqueeze(0)
    return img

def stylized_image_with_camera(inference_config, device, model, should_save=False):
    cap = cv.VideoCapture(0)

    k = 0
    while k != 27: # ESC for exit
        ret, frame = cap.read()
        optimizing_img = prepare_img_for_camera(frame)
        optimizing_img = model(optimizing_img)
        optimizing_img = optimizing_img.detach().cpu().numpy()[0]
        optimizing_img = post_process_image(optimizing_img)[:, :, ::-1]
        cv.imshow('camera', optimizing_img)
        k = cv.waitKey(5000)

inference_config = {}
inference_config['height'] = 500
inference_config['model_binaries_path'] = 'binaries'
inference_config['model_name'] = 'style_candy_datapoints_117380_cw_1.0_sw_30000.0_tw_0.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = prepare_model(inference_config, device)
stylized_image_with_camera(inference_config, device, model, should_save=False)
    
