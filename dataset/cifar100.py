import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None, download=False):          
        super(MyCIFAR100, self).__init__(root, train, transform, download)
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
            
        return img, target     
