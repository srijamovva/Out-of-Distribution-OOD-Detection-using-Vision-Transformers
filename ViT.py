import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import time
import copy

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
import timm
from torch.nn import CrossEntropyLoss
from transformers import ViTFeatureExtractor, ViTForImageClassification


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')


class ImageTransform:
    def __init__(self):
        pass

    def __call__(self, image):
        output = torch.from_numpy(feature_extractor(images=image.convert("RGB")).pixel_values[0])
        return output


def get_data_loaders(data_dir, batch_size, path , transform):
    data = datasets.ImageFolder(os.path.join(data_dir, path), transform = transform )
    loader = DataLoader(data, batch_size = batch_size, shuffle = True)

    return loader, len(data), data   

data_dir = "cifar100"

train_loader, train_data_len, train_data = get_data_loaders(data_dir, 128, "train/", transform = ImageTransform())
test_loader, test_data_len, test_data = get_data_loaders(data_dir, 128, "test/", transform = ImageTransform())
  