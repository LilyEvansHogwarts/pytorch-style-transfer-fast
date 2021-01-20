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

def stylize_static_image(inference_config, images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stylization_model = prepare_model(inference_config, device)

    for id, img in enumerate(images):
        optimizing_img = prepare_img(img, inference_config["height"], device)
        optimizing_img = stylization_model(optimizing_img)
        save_image(optimizing_img, os.path.join('output_images', str(id).zfill(8)+'.jpg'))

def mp4_to_jpg_sequence(video_file):
    vidcap = cv.VideoCapture(video_file)
    img_seq = []
    id = 0
    success, image = vidcap.read()
    while success:
        image_name = os.path.join('input_images', str(id).zfill(8)+'.jpg')
        cv.imwrite(image_name, image)
        img_seq.append(image_name)
        id += 1
        success, image = vidcap.read()

    return img_seq

def jpg_sequence_to_mp4(video_file, img_seq):
    frame = cv.imread(img_seq[0])
    height, width, channel = frame.shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(video_file, fourcc, 20.0, (width, height))

    for image in img_seq:
        video.write(cv.imread(image))

    video.release()
    cv.destroyAllWindows()



inference_config = {}
inference_config['height'] = 500
inference_config['model_binaries_path'] = 'binaries'
inference_config['model_name'] = 'style_candy_datapoints_117380_cw_1.0_sw_30000.0_tw_0.pth'

img_seq = mp4_to_jpg_sequence('56.MP4')
stylize_static_image(inference_config, img_seq)
img_seq = sorted([os.path.join('output_images', img) for img in os.listdir('output_images')])
jpg_sequence_to_mp4('output.mp4', img_seq)

    
