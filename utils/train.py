from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# TODO: check out other functions from the IPEO deep learning exercises (semantic segmentation and convnets), they might be useful too


class EarlyStopper:
    # from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    # adapted from CS-433 Machine Learning Exercises
    # Example training function for an epoch

    model.train()  # Important set model to train mode (affects dropout, batch norm etc)

    loss_history = []
    accuracy_history = []
    loss= 0
    progress_bar = tqdm(total=len(train_loader.torch_loader), desc=f"Loss: {loss:.5f}")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=device), target.to(device=device) #TODO target should be (#batchsize, H, W) for CELoss
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.set_description(f"Loss: {loss.item():.5f}")
        progress_bar.update(1)  # Manually update the progress bar
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        loss_history.append(loss.item())
        accuracy_history.append(correct / len(data))

        if batch_idx % (len(train_loader.torch_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss.item()/len(data):0.2e} batch_acc={correct/len(data):0.3f}"
            )

    return loss_history, accuracy_history


def run_mnist_training(model, num_epochs, lr, batch_size, device="cpu"):
    # adapted from CS-433 Machine Learning Exercises
    """Example training function, to be adapted."""
    # TODO: adapt to our data

    # ===== Data Loading =====
    # The input images should be normalized to have zero mean, unit variance
    # We could also add data augmentation here if we wanted
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)

    # Here we use the official test set as a validation set
    # This is not a good practice (but quite common since it is easier to setup)
    val_set = datasets.MNIST("./data", train=False, transform=transform)

    # The dataloaders can run in separate threads and handle the actual data
    # reading, augmenting and forming mini-batches
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # Can be important for training
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
    )

    # ===== Model, Optimizer and Criterion =====
    model = None  # OUr Model
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.functional.cross_entropy

    # ===== Train Model =====
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, epoch, criterion
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    # ===== Plot training curves =====
    n_train = len(train_acc_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_val = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(t_train, train_acc_history, label="Train")
    plt.plot(t_val, val_acc_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.figure()
    plt.plot(t_train, train_loss_history, label="Train")
    plt.plot(t_val, val_loss_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # ===== Plot low/high loss predictions on validation set =====
    points = get_predictions(
        model,
        device,
        val_loader,
        partial(torch.nn.functional.cross_entropy, reduction="none"),
    )
    points.sort(key=lambda x: x[1])
    plt.figure(figsize=(15, 6))
    for k in range(5):
        plt.subplot(2, 5, k + 1)
        plt.imshow(points[k][0].reshape(28, 28), cmap="gray")
        plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
        plt.subplot(2, 5, 5 + k + 1)
        plt.imshow(points[-k - 1][0].reshape(28, 28), cmap="gray")
        plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    # adapted from CS-433 Machine Learning Exercises
    model.eval()  # Important set model to eval mode (affects dropout, batch norm etc)
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() * len(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return test_loss, correct / len(val_loader.dataset)


@torch.no_grad()
def get_predictions(model, device, val_loader, criterion, num=None):
    # adapted from CS-433 Machine Learning Exercises
    model.eval()
    points = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)

        data = np.split(data.cpu().numpy(), len(data))
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points
