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
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel
softmax = nn.Softmax(dim=1)
from collections import OrderedDict
from scipy.stats import entropy
from torch.distributions import Categorical
from numpy import linalg as LA


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

class ImageTransform:
    def __init__(self):
        pass

    def __call__(self, image):
        output = torch.from_numpy(feature_extractor(images=image.convert("RGB")).pixel_values[0])
        return output


def get_data_loaders(data_dir, batch_size, path , transform, shuffle):
    data = datasets.ImageFolder(os.path.join(data_dir, path), transform = transform )
    loader = DataLoader(data, batch_size = batch_size, shuffle = shuffle)

    return loader, len(data)   

data_dir_train = "OOD_train_test/cifar10/"
data_dir_ood = "OOD_train_test/OOD_datasets"

# Cifar-10 train set
train_loader, train_data_len = get_data_loaders(data_dir_train, 128, "train/", transform = ImageTransform(), shuffle = True)

# Cifar-10 test set
cifar10_test_loader, cifar10_test_data_len = get_data_loaders(data_dir_train, 128, "test/", transform = ImageTransform(), shuffle = False)
# Cifar-100 train set 
cifar100_test_loader, cifar100_test_data_len = get_data_loaders(data_dir_ood, 128, "cifar100_train/", transform = ImageTransform(), shuffle = False)
# Food-101 train set
food101_test_loader, food101_test_data_len = get_data_loaders(data_dir_ood, 128, "food-101/", transform = ImageTransform(), shuffle = False)
#SVHN train set
svhn_test_loader, svhn_test_data_len = get_data_loaders(data_dir_ood, 128, "SVHN/", transform = ImageTransform(), shuffle = False)


test_loader = [cifar10_test_loader, cifar100_test_loader, food101_test_loader, svhn_test_loader]
test_data_len = [cifar10_test_data_len, cifar100_test_data_len, food101_test_data_len, svhn_test_data_len]
datasets = ['Cifar10-test', 'Cifar-100', 'Food-101', 'SVHN']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataloaders = {
    "train": train_loader,
    "test": test_loader
    
}

dataset_sizes = {
    "train": train_data_len,
    "test": test_data_len 
}




class ViT(torch.nn.Module):
    def __init__(self, num_labels):
        super(ViT, self).__init__()
        self.num_labels = num_labels
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.in_features = self.vit.pooler.dense.in_features
        self.num_labels = num_labels
        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_labels)
            )

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, inputs):
        cls_token = self.vit(inputs).last_hidden_state[:, 0]
        output = self.classifier(cls_token)
        output = self.softmax(output)

        return cls_token, output

    # def forward(self, inputs):
    #     tokens = self.vit(inputs).last_hidden_state
    #     mean_tokens = torch.mean(tokens, 1)
    #     output = self.classifier(mean_tokens)
    #     output = self.softmax(output)

    #     return mean_tokens, output



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the Pretrained Model
model = ViT(10)
model = model.to(device)
cost = CrossEntropyLoss()
cost = cost.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)


# Storing the CLS tokens of all the classes
def getClasstokens(classes_dict, tokens, labels):
    for i in range(len(tokens)):
        if labels[i] not in list(classes_dict.keys()):
            classes_dict[labels[i].item()] = []
        classes_dict[labels[i].item()].append(tokens[i])

    return classes_dict
        


# Training the Model on Cifar-100
def train_model(model, cost, optimizer, epochs):
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0
    classes_dict = {}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            cls_token, outputs = model(inputs)

            if epoch == epochs - 1:
                classes_dict = getClasstokens(classes_dict, cls_token, labels)
            _, predictions = torch.max(outputs, 1)
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(predictions == labels.data)


        train_loss = train_loss / dataset_sizes['train']
        train_acc = train_correct.double() / dataset_sizes['train']

        print("\nEpoch: {} Train Accuracy {:.4f} Train Loss {:.4f}".format(epoch, train_acc, train_loss))
    classes_dict = OrderedDict(sorted(classes_dict.items()))

    return model, classes_dict

model_trained, classes_dict = train_model(model, cost, optimizer, 30)
torch.save(model_trained, "cifar10_checkpoint_cls_30.pth")
torch.save(classes_dict, 'cifar10_classes_cls_30.pth')



model = torch.load('cifar10_checkpoint_cls_30.pth')
model = model.to(device)
classes_dict = torch.load('cifar10_classes_cls_30.pth')

print("Calculating Statistics")
# Calculate mean and covariance for each class
def GetStats(classes_dict):

    mean_dict = {}
    cov_inv_dict = {}
    for i in list(classes_dict.keys()):
        mean = np.array([float(sum(col))/len(col) for col in zip(*classes_dict[i])])
        mean = mean.reshape(1, mean.shape[0])
        mean_dict[i] = mean
        stack = torch.stack(classes_dict[i]).T
        cov = torch.cov(stack)
        cov_inv = torch.inverse(cov)
        cov_inv_dict[i] = cov_inv
  
    
    return mean_dict, cov_inv_dict
means_dict, cov_inv_dict = GetStats(classes_dict)
torch.save(means_dict, 'cifar10_mean_cls_30.pth')
torch.save(cov_inv_dict, 'cifar10_cov_inv_cls_30.pth')
print("Done\n")


# Out of distribution detection on Food-101 dataset

means = torch.load('cifar10_mean_cls_30.pth')
inv_cov = torch.load('cifar10_cov_inv_cls_30.pth')

# Calculating Mahalanobis Distance
def getDistance(token, means, inv_cov):
    distances = []

    for i in range(len(means)):
        inv_cov[i] = inv_cov[i].to("cpu")
        token = token.to("cpu")
        diff = torch.subtract(token, torch.from_numpy(means[i])).float()
        mul = torch.matmul(diff, inv_cov[i])
        dist = torch.matmul(mul, diff.T)
        distances.append(dist[0][0].item())
    return min(distances)

    
model.eval()

for data in range(len(dataloaders['test'])):

    outlier_count = 0
    not_outlier_count = 0
    test_acc = 0
    for inputs, labels in tqdm(dataloaders['test'][data]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        cls_tokens, outputs = model(inputs)
        max_softmax = []
        max_softmax = torch.max(outputs, dim = 1).values.tolist()
        max_softmax = [round(i, 2) for i in max_softmax]
        _,prediction = torch.max(outputs.data, 1) 
        test_acc += int(torch.sum(prediction == labels.data))


        # Calculate entropies for each class based on softmax outputs
        entropies = []
        outputs = outputs.to("cpu").detach().numpy()
        for i in range(outputs.shape[0]):
            entropies.append(entropy(outputs[i], base = 2))

        # Calculate Mahalanobis distance for each test point
        distances = []
        for i in cls_tokens:
            distances.append(getDistance(i, means, inv_cov))

        for i in range(len(max_softmax)):
            if max_softmax[i] < 0.5 or distances[i] > 2000 or entropies[i] > 1:
                outlier_count+= 1
            else:
                not_outlier_count+= 1

    test_acc = test_acc / test_data_len[data]

    print(datasets[data])
    print("Test Accuracy: {}  ".format(test_acc * 100))

    print("\nOutlier count: {} Outlier % {:.4f} ".format(outlier_count, (outlier_count / test_data_len[data]) * 100))
    print("\nIn Distribution count: {} In distribution % {:.4f} ".format(not_outlier_count, (not_outlier_count / test_data_len[data]) * 100))
    print("\n")

    


    


    

