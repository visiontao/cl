# Copyright 2022-present, Tao Zhuo
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys
import torch
import numpy as np
import torch.nn.functional as F

from torchvision.transforms import transforms
from torchvision import datasets
from datetime import datetime

from PIL import Image, ImageOps

from backbone.resnet18 import ResNet18
from backbone.mlp import MLP

from dataset.mnist_tranforms.permutation import Permutation
from dataset.mnist_tranforms.rotation import Rotation

from dataset.mnist import MyMNIST
from dataset.cifar10 import MyCIFAR10
from dataset.cifar100 import MyCIFAR100
from dataset.tinyimagenet import TinyImagenet


# initialize a Dataset
def get_dataset(root, data_name, train, transform, download=False):
    if 'mnist' in data_name:
        root = os.path.join(root, 'MNIST')
        dataset = MyMNIST(root, train, transform, download)
    elif data_name == 'cifar10':
        root = os.path.join(root, 'CIFAR10')
        dataset = MyCIFAR10(root, train, transform, download)
    elif data_name == 'cifar100':
        root = os.path.join(root, 'CIFAR100')
        dataset = MyCIFAR100(root, train, transform, download)
    elif data_name == 'tinyimg':
        root = os.path.join(root, 'TINYIMG')
        dataset = TinyImagenet(root, train, transform, download)

    return dataset

# filter the output space of the t-th task
def filter_classes(output, t, n_classes, n_tasks):    
    n_cpt = n_classes // n_tasks  # n classes per task
    
    min_label = n_cpt * t
    max_label = n_cpt * (t + 1)
    
    output[:, 0:min_label] = -float('inf')
    output[:, max_label:] = -float('inf')
    
    return output


# get t-th split from the dataset for Class-IL and Task-IL
def get_split(dataset, t, n_classes, n_tasks):    
    n_cpt = n_classes // n_tasks  # n classes per task
    
    min_label = n_cpt * t
    max_label = n_cpt * (t + 1)
    
    indices = []
    for i in range(len(dataset)):
        if dataset.targets[i] in range(min_label, max_label):
            indices.append(i)

    split = torch.utils.data.Subset(dataset, indices)
        
    return split

        
def get_backbone(data_name, n_classes):
    if data_name in ['cifar10', 'cifar100', 'tinyimg']:
        return ResNet18(n_classes)
    elif data_name in ['perm-mnist', 'rot-mnist']:
        return MLP(28 * 28, n_classes)


def get_transform(data_name, train=True):    
    if 'mnist' in data_name:        
        if train:
            if data_name == 'perm-mnist':
                transform = transforms.Compose([transforms.ToTensor(), Permutation()])                
            elif data_name == 'rot-mnist': 
                transform = transforms.Compose([Rotation(), transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
    else:
        if data_name == 'cifar10':
            size=(32, 32)
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2615)
        elif data_name == 'cifar100':
            size=(32, 32)
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)   
        elif data_name == 'tinyimg':
            size=(64, 64)
            mean = (0.4802, 0.4480, 0.3975)
            std = (0.2770, 0.2691, 0.2821)
            
        if train:     
            transform = transforms.Compose([
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),            
                transforms.ToTensor(),        
                transforms.Normalize(mean, std)])                                  
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    return transform
    
    
def transform_resize(dataset):
    if 'mnist' in dataset:
        transform = transforms.Compose([
            transforms.Grayscale(),        
            transforms.Resize(size=(28, 28))])        
    elif 'cifar' in dataset:          
        transform = transforms.Resize(size=(32, 32))
    elif dataset == 'tinyimg': 
        transform = transforms.Resize(size=(64, 64)) 
        
    transform = transforms.Compose([
        transform, transforms.ToTensor()])        
        
    return transform  

    
def progress_bar(batch_idx, max_iter, task_id, epoch, loss):
    """
    Prints out the progress bar on the stderr file.
    :param batch_idx: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param task_number: the task index
    :param loss: the current value of the loss function
    """
    if not (batch_idx + 1) % 10 or (batch_idx + 1) == max_iter:
        progress = min(float((batch_idx + 1) / max_iter), 1)
        progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
        print('\r[ {} ] Task {} | epoch {}: |{}| loss: {}'.format(
            datetime.now().strftime("%m-%d | %H:%M"),
            task_id,
            epoch,
            progress_bar,
            round(loss, 4)
        ), file=sys.stderr, end='', flush=True)
        