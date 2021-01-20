import torch
import torchvision
import os
import numpy as np
from collections import namedtuple
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

class SequentialSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, subset_size):
        assert isinstance(data_source, torch.utils.data.Dataset) or isinstance(data_source, torchvision.datasets.ImageFolder)
        self.data_source = data_source

        if subset_size is None:
            subset_size = len(data_source)
        assert 0 < subset_size <= len(data_source), f'Subset size should be between (0, {len(data_source)}'
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, target_width):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]

        h, w = load_image(self.img_paths[0]).shape[:2]
        img_height = int(h * target_width / w)
        self.target_width = target_width
        self.target_height = target_height

        self.transform = torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx], target_shape=(self.target_height, self.target_width))
        tensor = self.transform(img)
        return tensor

def calculate_content_loss(content_targets, content_features):
    content_loss = 0.0
    for t, f in zip(content_targets, content_features):
        content_loss += torch.nn.MSELoss(reduction='mean')(t, f)
    return content_loss

def calculate_style_loss(style_targets, style_features):
    style_loss = 0.0
    for gram_t, gram_f in zip(style_targets, style_features):
        style_loss += torch.nn.MSELoss(reduction='mean')(gram_t, gram_f)
    return style_loss

def calculate_tv_loss(optimizing_img):
    batch_size = optimizing_img.shape[0]
    tv_loss = torch.sum(torch.abs(optimizing_img[:, :, :, :-1] - optimizing_img[:, :, :, 1:])) + torch.sum(torch.abs(optimizing_img[:, :, :-1, :] - optimizing_img[:, :, 1:, :]))
    return tv_loss / batch_size

def gram_matrix(x, should_normalize=True):
    batch_size, ch, h, w = x.size()
    features = x.view(batch_size, ch, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'{img_path} does not exist.')

    img = cv.imread(img_path)[:, :, ::-1] # convert BGR to RGB

    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * new_height / current_height)
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32) # convert from uint8 to float32
    img /= 255 # get to [0, 1] range
    return img

def prepare_img(img_path, target_shape, device, batch_size=1, should_normalize=True, is_255_range=False):
    img = load_image(img_path, target_shape=target_shape)

    transform_list = [torchvision.transforms.ToTensor()]
    if is_255_range:
        transform_list.append(torchvision.transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transform.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else torchvision.transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = torchvision.transforms.Compose(transform_list)

    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)
    return img

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if p.requires_grad)

def post_process_image(dump_img):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy image got {type(dump_img)}'

    mean = IMAGENET_MEAN_1.reshape(-1, 1, 1)
    std = IMAGENET_STD_1.reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean
    dump_img = (np.clip(dump_img, 0, 1) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)
    return dump_img

def display_image(optimizing_img):
    outputs = optimizing_img.detach().cpu().numpy()
    print(outputs.shape)
    output_img = post_process_image(outputs[0])
    plt.imshow(output_img)
    plt.show()

def save_image(optimizing_img, image_file_name):
    output = optimizing_img.detach().cpu().numpy()
    output_img = post_process_image(output[0])
    output_img = output_img[:, :, ::-1]
    cv.imwrite(image_file_name, output_img)

def get_training_data_loader(training_config, should_normalize=True, is_255_range=False):
    transform_list = [torchvision.transforms.Resize(training_config['image_size']),
                      torchvision.transforms.CenterCrop(training_config['image_size']),
                      torchvision.transforms.ToTensor()]

    if is_255_range:
        transform_list.append(torchvision.transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(torchvision.transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else torchvision.transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = torchvision.transforms.Compose(transform_list)

    train_dataset = torchvision.datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    training_config['subset_size'] = len(sampler)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler, drop_last=True)
    print(f'Using {len(train_loader) * training_config["batch_size"] * training_config["num_of_epochs"]} datapoints ({len(train_loader) * training_config["num_of_epochs"]} batches) (MS COCO images) for transformer network training.')
    return train_loader

def get_training_metadata(training_config):
    num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    training_metadata = {
        'content_weight': training_config['content_weight'],
        'style_weight': training_config['style_weight'],
        'tv_weight': training_config['tv_weight'],
        'num_of_datapoints': num_of_datapoints
    }
    return training_metadata

def print_model_metadata(training_state):
    print('Model training metadata:')
    for key, value in training_state.items():
        if key != 'state_dict' and key != 'optimizer_state':
            print(key, ':', value)


