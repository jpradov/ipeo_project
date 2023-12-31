import os

import kornia.augmentation as K
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from skimage import exposure
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
from utils.data import normalized_image

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, device, val_loader, criterion):
    """ Function to evaluate current model given a data_loader.
    adapted from CS-433 Machine Learning Exercises 
    """
    
    model.eval()

    # initialise counters
    test_loss = 0
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        test_loss += criterion(output, target).item() * len(data)
        pred = output.argmax(dim=1)

        true_pos += ((target == 1) & (pred == 1)).sum().item()
        true_neg += ((target == 0) & (pred == 0)).sum().item()
        false_neg += ((target == 1) & (pred == 0)).sum().item()
        false_pos += ((target == 0) & (pred == 1)).sum().item()

    # get final loss across full epoch
    test_loss /= len(val_loader.torch_loader.dataset)

    # Calculate metrics (on GPU if needed)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_pos) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    jaccard = true_pos / (true_pos + false_neg + false_pos) if (true_pos + false_neg + false_pos) else 0

    return test_loss, accuracy, jaccard, precision, recall, f1

def show_overlay(image, mask, prediction):
    """ 
    Function to overlay a prediction and ground truth mask over image and display figure
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 12))

    # reorder images to (H, W, C) and normalize image for better plotting
    image = torch.permute(image, (1, 2, 0)).cpu().numpy()
    image = normalized_image(image)
    mask = torch.permute(mask, (1, 2, 0)).cpu().numpy()
    prediction = torch.permute(prediction, (1, 2, 0)).cpu().numpy()

    # print mask and original
    axes[0].imshow(image)
    axes[0].imshow(mask, alpha=0.4,)
    axes[0].axis("off")
    axes[0].set_title("Ground Truth")

    # print prediction and original
    axes[1].imshow(image)
    axes[1].imshow(prediction, alpha=0.4,)
    axes[1].axis("off")
    axes[1].set_title("Predicted Mask")
    return

def visualise_batch_predictions(batch_image, batch_mask, batch_prediction, rescale=False, bands=[0, 1, 2]):
    """ 
    Function taking in image batch, mask batch and corresponding predictions and plots the overlay.

    batch_image (torch.tensor): tensor of image
    batch_mask (torch.tensor): tensor of mask
    batch_predictions (torch.tensor): tensor of predictions
    rescale (bool): set to True to undo any normalization done by the data loaders for better plotting.
    bands list(int): list of the inputs bands present in the image
    """

    # extract number of images
    batch_size = batch_image.shape[0]
    print("Visualizing {} examples".format(batch_size))
   
    # constants to rescale image
    means = -1 * torch.tensor([265.7371, 445.2234, 393.7881, 2773.2734, 0.8082])
    stds = 1 / torch.tensor([91.8786, 110.0122, 191.7516, 709.2327, 1.0345e-01])
    temp1, temp2 = torch.tensor([0, 0, 0, 0, 0]), torch.tensor([1, 1, 1, 1, 1])

    means, stds =  means[bands], stds[bands]
    temp1, temp2 = temp1[bands], temp2[bands]

    for index in range(batch_size):
        
        # get image, mask and prediction at index
        image = batch_image[index, :, :, :]
        mask = batch_mask[index, :, :, :]
        prediction = batch_prediction[index, :, :, :]
        
        # revert normalization
        if rescale:
            invTrans = transforms.Compose([transforms.Normalize(mean = temp1,
                                                            std = stds),
                                        transforms.Normalize(mean = means,
                                                            std = temp2),
                                    ])
            image = invTrans(image)

        # grab first three channels of image for plotting
        image = image[:3, :, :]

        # plot overlay
        show_overlay(image, mask, prediction)
