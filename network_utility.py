import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss for addressing class imbalance.

        Args:
            alpha (float, list, or tensor): Balancing factor for classes. If it's a tensor, it should be the same length as the number of classes.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = alpha.clone().detach().float()
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Convert alpha to be on the same device as inputs
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]  # Select the alpha value for each target class
        else:
            alpha_t = 1.0  # No class weighting

        # Probability of the true class
        pt = torch.exp(-ce_loss)

        # Compute the focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_alpha_inverse_frequency(class_distribution, eta):
    """
    Compute alpha values based on the inverse frequency of classes.

    Args:
        class_distribution (dict): Dictionary where keys are class IDs and values are the number of images per class.

    Returns:
        torch.Tensor: Alpha values as a tensor, where each value corresponds to the inverse frequency of a class.
    """
    total_images = sum(class_distribution.values())
    num_classes = len(class_distribution)

    # Calculate alpha as inverse frequency
    alpha = torch.tensor([np.sqrt(total_images / (num_classes * count)) for count in class_distribution.values()], dtype=torch.float32)
    fixed_alpha = 0.5
    alpha_interp = eta * alpha + (1 - eta) * fixed_alpha

    return alpha_interp


"""def compute_alpha_inverse_frequency(distribution, mode="make"):
    if mode == "make":
        total = sum(distribution.values())
        alpha = {label: total / count for label, count in distribution.items()}
        # Normalize alpha values
        sum_alpha = sum(alpha.values())
        alpha = {label: value / sum_alpha for label, value in alpha.items()}
        alpha = torch.tensor([alpha[label] for label in sorted(alpha.keys())])
        return alpha

    elif mode == "make_model":
        make_distribution, model_distribution = distribution
        total_make = sum(make_distribution.values())
        total_model = sum(model_distribution.values())

        make_alpha = {label: total_make / count for label, count in make_distribution.items()}
        model_alpha = {label: total_model / count for label, count in model_distribution.items()}

        # Normalize alpha values
        sum_make_alpha = sum(make_alpha.values())
        sum_model_alpha = sum(model_alpha.values())

        make_alpha = {label: value / sum_make_alpha for label, value in make_alpha.items()}
        model_alpha = {label: value / sum_model_alpha for label, value in model_alpha.items()}

        make_alpha = torch.tensor([make_alpha[label] for label in sorted(make_alpha.keys())])
        model_alpha = torch.tensor([model_alpha[label] for label in sorted(model_alpha.keys())])

        return make_alpha, model_alpha"""

class EarlyStopping:
    def __init__(self, patience=2, path='best_checkpoint.pt'):
        """
        Early stopping to stop the training when the loss does not improve after a certain number of epochs.

        Args:
            patience (int): How long to wait after the last time validation loss improved.
                            Default: 5
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if os.path.exists(self.path):
           os.remove(self.path)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def plot_loss_accuracy_curves(train_losses, val_losses, train_accuracies, val_accuracies, run_index=None):
    """
    Plots the training and validation loss/accuracy curves.
    
    Parameters:
    - train_losses (list): List of training losses recorded after each epoch.
    - val_losses (list): List of validation losses recorded after each epoch.
    - train_accuracies (list): List of training accuracies recorded after each epoch.
    - val_accuracies (list): List of validation accuracies recorded after each epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_losses, 'o-', label='Training Loss', color='tab:blue')
    ax1.plot(epochs, val_losses, 'o-', label='Validation Loss', color='tab:orange')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs, train_accuracies, 's-', label='Training Accuracy', color='tab:green')
    ax2.plot(epochs, val_accuracies, 's-', label='Validation Accuracy', color='tab:red')
    ax2.legend(loc='upper right')

    plt.title('Training and Validation Loss/Accuracy')

    if run_index==None:
      plt.savefig('loss_accuracy_curve.png')
    else:
      plt.savefig(f'loss_accuracy_curve{run_index}.png')
    
    plt.close()




