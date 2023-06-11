import numpy as np
from PIL import Image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None, download=False):   
        super(MyMNIST, self).__init__(root, train, transform, download)
        self.transform = transform
            
    def __getitem__(self, index):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target  