import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

from PIL import Image
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

# Utils
def show_sample(image, mask=None, alpha=0.7):
    print('Image shape:', image.shape)
    plt.imshow(image.permute(1, 2, 0))
    if mask is not None:
        print('Mask shape:', mask.shape)

        plt.imshow(mask[0], alpha=alpha)
    plt.show()

def show_samples(images, masks=None, alpha=0.7, nrow=4):
    print('Images shape:', images.shape)
    if masks is not None:
        print('Masks shape:', masks.shape)
        B, C, H, W = images.shape
        col = [0.2, 0.3, 0.8]
        col = torch.tensor(col).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(B, 1, H, W)
        images = torch.where(masks.repeat(1, 3, 1, 1) > 0,
                             alpha * col + (1 - alpha) * images, images)
    image_grid = make_grid(images, nrow=nrow, padding=0)
    plt.figure(figsize=(15, 15))
    plt.imshow(image_grid.permute(1, 2, 0), aspect='auto')
    plt.axis(False)
    plt.show()

class Denorm(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array([0.0, 0.0, 0.0]) if mean is None else mean
        self.std = np.array([1.0, 1.0, 1.0]) if std is None else std

    def __call__(self, x):
        """
        Denormalize the image.
        Args:
            x: tensor of shape [bs, c, h, w].
        Output:
            x_denorm: tensor of shape [bs, c, h, w].
        """
        denorm_fn = transforms.Normalize(mean=- self.mean / (self.std + 1e-8), std=1.0 / (self.std + 1e-8))
        x_denorm = []
        for x_i in x:
            x_denorm += [denorm_fn(x_i)]
        x_denorm = torch.stack(x_denorm, 0)
        return x_denorm

class GTEADataset(Dataset):
    """
        GTEA dataset from http://cbs.ic.gatech.edu/fpv.
        Images and masks of dims:
        - GTEA: [405, 720]
        - GTEA_GAZE_PLUS: [720, 960]
    """
    def __init__(self, data_base_path, partition, image_transform=None,
                 mask_transform=None, seed=1234):
        super(GTEADataset, self).__init__()
        self.data_base_path = data_base_path
        self.partition = partition
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.seed = seed
        self.image_paths, self.mask_paths = self._get_paths()

    def _get_paths(self):

        # Image paths
        image_paths = []

        # GTEA
        image_names = sorted(os.listdir(os.path.join(self.data_base_path, 'GTEA', 'Images')))
        image_paths = image_paths + [os.path.join(self.data_base_path, 'GTEA', 'Images', f) for f in image_names]

        # GTEA_GAZE_PLUS
        for folder in sorted(os.listdir(os.path.join(self.data_base_path, 'GTEA_GAZE_PLUS', 'Images'))):
            image_names = sorted(os.listdir(os.path.join(self.data_base_path, 'GTEA_GAZE_PLUS', 'Images', folder)))
            image_paths = image_paths + [os.path.join(self.data_base_path, 'GTEA_GAZE_PLUS', 'Images', folder, f) for f in image_names]

        # Mask paths
        mask_paths = [f.replace('Images', 'Masks').replace('.jpg', '.png') for f in image_paths]

        # Split data
        num_samples = len(image_paths)
        num_train = int(np.round(0.6 * num_samples))
        num_validation = int(np.round(0.2 * num_samples))
        num_test = int(np.round(0.2 * num_samples))
        idxs = np.arange(num_samples)
        np.random.seed(self.seed)
        np.random.shuffle(idxs)
        idxs_train = idxs[:num_train]
        idxs_validation = idxs[num_train:num_train + num_validation]
        idxs_test = idxs[-num_test:]
        if self.partition in ['train', 'training']:
            image_paths = [image_paths[i] for i in idxs_train]
            mask_paths = [mask_paths[i] for i in idxs_train]
        elif self.partition in ['val', 'validation', 'validating']:
            image_paths = [image_paths[i] for i in idxs_validation]
            mask_paths = [mask_paths[i] for i in idxs_validation]
        elif self.partition in ['test', 'testing']:
            image_paths = [image_paths[i] for i in idxs_test]
            mask_paths = [mask_paths[i] for i in idxs_test]
        else:
            raise Exception(f'Error. Partition "{self.partition}" is not supported.')
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load image and mask
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        # Transforms
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return image, mask



def get_dataloader(data_base_path, partition,image_transform=None,
                   mask_transform=None, batch_size=16, num_workers=0, pin_memory=True, shuffle=False):
    """
    Get the dataloader.
    Args:
        data_base_path: string where the data are stored.
        partition: string in ['train', 'validation', 'test'].
        datasets: list of strings for selecting the sounrce of the data.
        image_transforms: transform applied to the image.
        mask_transform: transform applied to the mask.
        batch_size: integer that specifies the batch size.
        num_workers: the number of workers.
        pin_memory: boolean.

    Output:
        dl: the dataloader (PyTorch DataLoader).
    """

    tranform = transforms.Compose([
        transforms.Resize((405, 720)),
        transforms.ToTensor(),
    ])
    ds_gtea = GTEADataset(
        data_base_path=os.path.join(data_base_path, 'hand2K_dataset'),
        partition=partition,
        image_transform=tranform if image_transform is None else image_transform,
        mask_transform=tranform if mask_transform is None else mask_transform,
        )

    dl = DataLoader(ds_gtea, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dl


##################################################
# Debug
##################################################

if __name__ == '__main__':
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        lambda m: torch.where(m > 0, torch.ones_like(m), torch.zeros_like(m))
    ])
    dl_args = {
        'data_base_path': './data',
        'partition': 'validation',
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'batch_size': 16,
    }
    dl = get_dataloader(**dl_args)