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
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score

@torch.no_grad()
def evaluate(model, device, val_loader, criterion):
    # adapted from CS-433 Machine Learning Exercises
    model.eval()
    test_loss = 0
    correct = 0
    target_tot = []
    pred_tot = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # Assuming this is a pixel-wise classification

        target_tot.extend(target.view(-1).cpu().numpy())
        pred_tot.extend(pred.view(-1).cpu().numpy())
        test_loss += criterion(output, target).item() * len(data)

    test_loss /= len(val_loader.torch_loader.dataset)

    # Calculate metrics using sklearn
    accuracy = accuracy_score(target_tot, pred_tot)   
    jaccard = jaccard_score(target_tot, pred_tot, average='macro') # equivalent to IOU
    precision = precision_score(target_tot, pred_tot, average='macro')
    recall = recall_score(target_tot, pred_tot, average='macro')
    f1 = f1_score(target_tot, pred_tot, average='macro') # equivalent to Dice index

    return test_loss, accuracy, jaccard, precision, recall, f1

def show_overlay(image, mask, prediction, rescale=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 12))

    image = torch.permute(image, (1, 2, 0)).cpu().numpy()
    image = normalized_image(image)
    mask = torch.permute(mask, (1, 2, 0)).cpu().numpy()
    prediction = torch.permute(prediction, (1, 2, 0)).cpu().numpy()

    axes[0].imshow(image)
    axes[0].imshow(mask, alpha=0.4,)
    axes[0].axis("off")
    axes[0].set_title("Ground Truth")

    axes[1].imshow(image)
    axes[1].imshow(prediction, alpha=0.4,)
    axes[1].axis("off")
    axes[1].set_title("Predicted Mask")
    return

def show_results_tensor(batch_image, batch_mask, batch_prediction, rescale=False, bands=[0, 1, 2]):
    batch_size = batch_image.shape[0]
    
    print("Visualsing {} examples".format(batch_size))

    batch_prediction = batch_prediction.sigmoid() > 0.5 # take sigmoid and threshold at 0.5
    
    # constants to rescale image
    means = -1 * torch.tensor([265.7371, 445.2234, 393.7881, 2773.2734])
    stds = 1 / torch.tensor([91.8786, 110.0122, 191.7516, 709.2327])
    temp1 = torch.tensor([0, 0, 0, 0])
    temp2 = torch.tensor([1, 1, 1, 1])

    means, stds =  means[bands], stds[bands]
    temp1, temp2 = temp1[bands], temp2[bands]

    for index in range(batch_size):
        
        image = batch_image[[index], :, :, :].squeeze(0)
        mask = batch_mask[[index], :, :, :].squeeze(0)
        prediction = batch_prediction[[index], :, :, :].squeeze(0)
        
        print("Proportion of Positive Pixels predicted: ", prediction.float().mean())
        print("Proportion of Positive Pixels in Mask: ", mask.float().mean())

        if rescale:
            invTrans = transforms.Compose([ transforms.Normalize(mean = temp1,
                                                            std = stds),
                                        transforms.Normalize(mean = means,
                                                            std = temp2),
                                    ])
            image = invTrans(image)

        show_overlay(image, mask, prediction, rescale=rescale)
