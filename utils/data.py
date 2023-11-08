"""This module contains all function related to data loading and processing. """

import numpy as np
from skimage import exposure
from torchgeo.datasets.geo import NonGeoDataset
from torch.utils.data import Dataset
import config
import os
import torch
from skimage.io import imread

def normalized_image(image):
    """
    Rescale image to values between 0 to 255 (capping outlier values)

    Parameters
    ==================
    image: Numpy array
        Image numpy array with shape (height, width, num_bands)

    percentiles: tuple
        Tuple of min and max percentiles to cap outlier values

    Returns
    ==================
    output: Numpy array
        Normalized image numpy array

    """
    output = np.zeros_like(image)
    for k in range(image.shape[2]):  # for each band
        p_min, p_max = np.min(image[:, :, k]), np.max(image[:, :, k])
        output[:, :, k] = exposure.rescale_intensity(
            image[:, :, k], in_range=(p_min, p_max), out_range=(0, 255)
        )
    return output.astype(np.uint8)

class PlanetDataset(NonGeoDataset,Dataset):
    def __init__(self, data_dir=config.PATH_TO_DATA, bands=[0,1,2,3], transform=None, target_transform=None):
        """
        Args:
            data_dir (str): The root directory where the Planet dataset is stored.
            bands (list(int)) : A list of band indexes to be used.
            transform (callable, optional): A function/transform to apply to the data samples.
            target_transform (callable, optional): A function/transform to apply to the target masks.
        """
        
        self.root = data_dir
        self.image_path = os.path.join(self.root, r"images")
        self.mask_path = os.path.join(self.root, r"labels")
        self.folder_data = os.listdir(self.image_path)
        self.folder_mask = os.listdir(self.mask_path)
        self.bands=bands
        self.transform = transform
        self.target_transform = target_transform
        # Map that converts image names to indexes in the dataset
        self.id2index = { int(image_id[:-4]) : i for i, image_id in enumerate(self.folder_data)}

    def __getitem__(self, index):
        sample_path = os.path.join(self.image_path,self.folder_data[index])
        mask_path = os.path.join(self.mask_path,self.folder_mask[index])

        sample = torch.from_numpy(self.custom_loader(path=sample_path)[:, :, self.bands])
        sample = torch.permute(sample,(2,0,1))

        mask = torch.from_numpy(self.custom_loader(mask_path))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sample, mask

    def __len__(self):

        return len(self.folder_data)
    
    def custom_loader(self,path):  
        return imread(path).astype(np.float32) 
    

        
    def __str__(self):
        
        return f"PlanetDataset(root={self.root}, num_samples={len(self)}, transform={self.transform}, target_transform={self.target_transform})"