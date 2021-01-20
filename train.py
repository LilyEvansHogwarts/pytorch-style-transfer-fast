import torch
import torchvision
import os
import numpy as np
from collections import namedtuple
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model import PerceptualLossNet, TransformerNet
from utils import *

def train(training_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)
    transformer_net = TransformerNet().train().to(device)

    optimizer = torch.optim.Adam(transformer_net.parameters(), lr=training_config['learning_rate'])
    style_img_path = os.path.join(training_config['style_images_dir'], training_config['style_image_name'])
    style_image = prepare_img(style_img_path, training_config['height'], device)
    features_for_style_image = perceptual_loss_net(style_image)
    style_targets = features_for_style_image.style_features

    train_loader = get_training_data_loader(training_config)
    writer_stylized = SummaryWriter(f'runs/stylized')
    writer_real = SummaryWriter(f'runs/real')

    step = 0
    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # step1: Calculate the content targets
            content_batch = content_batch.to(device)
            features_for_content_batch = perceptual_loss_net(content_batch)
            content_targets = features_for_content_batch.content_features

            # step2: Calculate the content_features and style_features for stylized_content_batch
            stylized_content_batch = transformer_net(content_batch)
            features_for_stylized_content_batch = perceptual_loss_net(stylized_content_batch)
            content_features = features_for_stylized_content_batch.content_features
            style_features = features_for_stylized_content_batch.style_features

            # step3: calculate content_loss, style_loss, tv_loss
            content_loss = training_config['content_weight'] * calculate_content_loss(content_targets, content_features) 
            style_loss = training_config['style_weight'] * calculate_style_loss(style_targets, style_features)
            tv_loss = training_config['tv_weight'] * calculate_tv_loss(stylized_content_batch)

            # step4: calculate total_loss
            total_loss = content_loss + style_loss + tv_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad() # clear gradients for the next round

            if batch_id % 20 == 0:
                print(f'Epochs: {epoch} | total_loss: {total_loss.item():12.4f} | content_loss: {content_loss.item():12.4f} | style_loss: {style_loss.item():12.4f} | tv_loss: {tv_loss.item():12.4f}')
                
                display_image(content_batch)
                display_image(stylized_content_batch)
                
                with torch.no_grad():
                    image_grid_stylized = torchvision.utils.make_grid(stylized_content_batch, normalize=True)
                    image_grid_real = torchvision.utils.make_grid(content_batch, normalize=True)
                    writer_stylized.add_image('stylized image', image_grid_stylized, global_step=step)
                    writer_real.add_image('original image', image_grid_real, global_step=step)

                training_state = get_training_metadata(training_config)
                training_state['state_dict'] = transformer_net.state_dict()
                training_state['optimizer_state'] = optimizer.state_dict()
                model_name = f'style_{training_config["style_image_name"].split(".")[0]}_datapoints_{training_state["num_of_datapoints"]}_cw_{str(training_config["content_weight"])}_sw_{str(training_config["style_weight"])}_tw_{str(training_config["tv_weight"])}.pth'
                torch.save(training_state, os.path.join(training_config["model_binaries_path"], model_name))

                


            

training_config = {}
training_config['content_weight'] = 1e0
training_config['style_weight'] = 3e4
training_config['tv_weight'] = 0
training_config['num_of_epochs'] = 10
training_config['subset_size'] = None
training_config['style_image_name'] = 'candy.jpg'
training_config['height'] = 500
training_config['style_images_dir'] = 'data/style_images'
training_config['image_size'] = 256
training_config['batch_size'] = 8
training_config['learning_rate'] = 1e-4
training_config['dataset_path'] = 'dataset/ms_coco'
training_config['model_binaries_path'] = 'binaries'

train(training_config)

