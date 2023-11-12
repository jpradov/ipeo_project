"""This module contains all function related to data loading and processing. """

import os

import kornia.augmentation as K
import numpy as np
import pandas as pd
import torch
from skimage import exposure
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset, Subset

import config


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


class PlanetBaseDataset(Dataset):
    """Base Dataset Class to load in all images with the requested bands."""

    def __init__(self, data_dir=config.PATH_TO_DATA, bands=[0, 1, 2, 3]):
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
        self.bands = bands

    def __getitem__(self, index):
        sample_path = os.path.join(self.image_path, self.folder_data[index])
        mask_path = os.path.join(self.mask_path, self.folder_mask[index])

        sample = torch.from_numpy(
            self.custom_loader(path=sample_path)[:, :, self.bands]
        )
        mask = torch.from_numpy(self.custom_loader(mask_path))

        # permute channels in the required order
        sample = torch.permute(sample, (2, 0, 1))

        # TODO: decide on normalization of satellite images!
        # min max normalization is difficult given quite a few outliers, standardization suffers from similar problems.
        """ 
        # mins, max of training raw images.
        mins = torch.tensor([ 0., 21.,  6., 77.])
        maxs = torch.tensor([ 4433.,  5023.,  8230., 10000.])

        # access relevant channels
        mins, maxs = mins[self.bands], maxs[self.bands]

        # normalization to 0-1 range 
        # sample = (sample - mins) / (maxs - mins)
        means = torch.tensor([ 265.7371,  445.2234,  393.7881, 2773.2734])
        stds = torch.tensor([ 91.8786, 110.0122, 191.7516, 709.2327])
        means, stds = means[self.bands], stds[self.bands]

        # sample = (sample - means) / stds
        """
        return sample, mask

    def __len__(self):
        return len(self.folder_data)

    def custom_loader(self, path):
        return imread(path).astype(np.float32)

    def __str__(self):
        return f"PlanetDataset(root={self.root}, num_samples={len(self)}, transform={self.transform}, target_transform={self.target_transform})"


class CustomDataLoader:
    """warpper class around regular PyTorch dataloader to be able to apply Kornia's batch transforms efficiently"""

    def __init__(
        self,
        dataset,
        transform,
        batch_size=4,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    ):
        self.torch_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
        )
        self.transforms = transform

    def __iter__(
        self,
    ):
        # get the batches of the original loader
        for sample_batch, mask_batch in self.torch_loader:
            # apply the transform
            mask_batch = mask_batch[:, None, :, :]
            sample_batch, mask_batch = self.transforms(sample_batch, mask_batch)
            # yield current batch to the iterator
            yield sample_batch, mask_batch.squeeze().type(torch.LongTensor) #BCELoss requires size to be (batch, H, W) and dtype LongTensor


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    bands: list = [0, 1, 2, 3],
    batch_transforms: bool = None,
    num_workers: int = 0,
):
    """Function to create individual dataloaders for test, train and val subset, including data augmentations"""
    # Create base datasets
    dataset = PlanetBaseDataset(data_dir, bands)

    # Load predefined splits
    splits = pd.read_csv(os.path.join(data_dir, "data_split.csv"))

    # extract train, test, val splits
    train_indices = splits[splits["split"] == "Train"].index.values
    test_indices = splits[splits["split"] == "Test"].index.values
    val_indices = splits[splits["split"] == "Validation"].index.values

    means = torch.tensor([265.7371, 445.2234, 393.7881, 2773.2734])
    stds = torch.tensor([91.8786, 110.0122, 191.7516, 709.2327])
    means, stds = means[bands], stds[bands]

    # define transforms
    train_transforms = K.container.AugmentationSequential(
        K.Resize((224, 224)),
        K.RandomResizedCrop(size=(224, 224), p=0.5),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomBoxBlur(
            kernel_size=(20, 20), border_type="reflect", normalized=True, p=0.2
        ),  # Note that the randomblur is applied to the image but no the mask!
        # K.RandomGrayscale(p=0.2), # only works when we have 3 input channels
        K.Normalize(mean=means, std=stds),
        same_on_batch=batch_transforms,
        data_keys=["image", "mask"],
    )

    test_transforms = K.container.AugmentationSequential(
        K.Resize((224, 224)),
        K.Normalize(mean=means, std=stds),
        same_on_batch=batch_transforms,
        data_keys=["image", "mask"],
    )

    # create data subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # create dataloaders for each subset
    train_dataloader = CustomDataLoader(
        train_dataset,
        transform=train_transforms,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
    )
    val_dataloader = CustomDataLoader(
        val_dataset,
        transform=test_transforms,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
    )
    test_dataloader = CustomDataLoader(
        test_dataset,
        transform=test_transforms,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader


# Old way of just loading all the images - left in as reference
class PlanetDataset(Dataset):
    def __init__(
        self,
        data_dir=config.PATH_TO_DATA,
        bands=[0, 1, 2, 3],
        transform=None,
        target_transform=None,
    ):
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
        self.bands = bands
        self.transform = transform
        self.target_transform = target_transform
        # Map that converts image names to indexes in the dataset
        self.id2index = {
            int(image_id[:-4]): i for i, image_id in enumerate(self.folder_data)
        }

    def __getitem__(self, index):
        sample_path = os.path.join(self.image_path, self.folder_data[index])
        mask_path = os.path.join(self.mask_path, self.folder_mask[index])

        sample = torch.from_numpy(
            self.custom_loader(path=sample_path)[:, :, self.bands]
        )
        sample = torch.permute(sample, (2, 0, 1))

        mask = torch.from_numpy(self.custom_loader(mask_path))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sample, mask

    def __len__(self):
        return len(self.folder_data)

    def custom_loader(self, path):
        return imread(path).astype(np.float32)

    def __str__(self):
        return f"PlanetDataset(root={self.root}, num_samples={len(self)}, transform={self.transform}, target_transform={self.target_transform})"
