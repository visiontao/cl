import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root, train=True, transform=None, download=False):
        self.transform = transform
        
        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(root, 'processed/x_%s_%02d.npy' % 
                                    ('train' if train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(root, 'processed/y_%s_%02d.npy' % 
                                    ('train' if train else 'val', num+1))))            
        self.targets = np.concatenate(np.array(self.targets))
        
#         print ('ledsfaf', np.unique(self.targets))
                            
#         mask = np.zeros(self.data.shape[0], dtype=bool)
#         mask[list(range(0, self.data.shape[0], 2))] = True

#         self.data = self.data[mask]     
#         self.targets = self.targets[mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
            
