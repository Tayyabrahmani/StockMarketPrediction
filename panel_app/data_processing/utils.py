import os
import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, learning_rate, lradj='type1'):
    """
    Adjusts the learning rate based on the epoch and learning rate adjustment type.

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate needs to be adjusted.
        epoch (int): Current epoch number.
        learning_rate (float): Initial learning rate.
        lradj (str): Type of learning rate adjustment ('type1', 'type2', or other).
    """
    if lradj == 'type1':
        lr_adjust = {
            2: learning_rate * 0.5 ** 1,
            4: learning_rate * 0.5 ** 2,
            6: learning_rate * 0.5 ** 3,
            8: learning_rate * 0.5 ** 4,
            10: learning_rate * 0.5 ** 5,
        }
    elif lradj == 'type2':
        lr_adjust = {
            5: learning_rate * 0.5 ** 1,
            10: learning_rate * 0.5 ** 2,
            15: learning_rate * 0.5 ** 3,
            20: learning_rate * 0.5 ** 4,
            25: learning_rate * 0.5 ** 5,
        }
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.

        Parameters:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Checks whether the validation loss has improved, and handles early stopping logic.

        Parameters:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save when validation loss improves.
            path (str): Path to save the checkpoint.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Saves the model when validation loss decreases.

        Parameters:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save.
            path (str): Path to save the checkpoint.
        """
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model checkpoint
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)  # Save to the specified path
        self.val_loss_min = val_loss
