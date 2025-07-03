

import random
import torch
import os
import math
import warnings
from matplotlib import pyplot as plt
import random
from scipy.stats import pearsonr
import numpy as np
import anndata as ad

from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def split_adata_random(adata,datasplit,xmax,ymax):
    trian_split = []
    for p in adata.obsm['spatial']:
        x,y = p
        is_train = random.random() <= datasplit
        if x>=0 and y>=0 and x<=xmax and y<=ymax:
            if is_train:
                trian_split.append('train')
            else:
                trian_split.append('valid')
        else:
            trian_split.append('error')

    adata.obs['dataset']=trian_split
    train_adata = adata[adata.obs['dataset'] == 'train'].copy()
    valid_adata = adata[adata.obs['dataset'] == 'valid'].copy()
    return train_adata,valid_adata


class Log():
    def __init__(self, log_dir,fold=None):
        self.log_dir            = log_dir
        if fold:
            self.save_path          = os.path.join(self.log_dir,fold)
        else:
            self.save_path = log_dir
        self.iter_loss = []
        self.losses             = []
        self.val_loss           = []
        self.pcc = []
        self.val_pcc = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def append_iter_loss(self,loss):
        self.iter_loss.append(loss)
        with open(os.path.join(self.save_path, "iter_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        
        if(len(self.iter_loss)%100 ==0):
            self.iter_loss_plot()


    def iter_loss_plot(self):
        iters = range(len(self.iter_loss))

        plt.figure()
        plt.plot(iters, self.iter_loss, 'blue', linewidth = 2, label='train loss')
        plt.grid(True)
        plt.xlabel('iteration')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_path, "iteration_loss.png"))
        plt.cla()
        plt.close("all")

    def append_epoch_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_epoch_plot()

    def append_epoch_pcc(self, pcc, val_pcc):
        self.pcc.append(pcc)
        self.val_pcc.append(val_pcc)
        with open(os.path.join(self.save_path, "epoch_pcc.txt"), 'a') as f:
            f.write(str(pcc))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_pcc.txt"), 'a') as f:
            f.write(str(val_pcc))
            f.write("\n")
        self.pcc_epoch_plot()

    def pcc_epoch_plot(self):
        iters = range(len(self.pcc))

        plt.figure()
        plt.plot(iters, self.pcc, 'blue', linewidth = 2, label='train pcc')
        plt.plot(iters, self.val_pcc, 'cornflowerblue', linewidth = 2, label='val pcc')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.save_path, "epoch_pcc.png"))

        plt.cla()
        plt.close("all")

    def loss_epoch_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'blue', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'cornflowerblue', linewidth = 2, label='val loss')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

def calculate_pcc(target, output):
    batch_size, num_samples = target.shape
    pccs = []

    for i in range(batch_size):
        current_target = target[i].cpu().detach().numpy()
        current_output = output[i].cpu().detach().numpy()
        pcc, _ = pearsonr(current_target, current_output)
        pccs.append(pcc)
    
    avg_pcc = np.mean(pccs)
    
    return avg_pcc

def calculate_pmse(target, output):

    current_target = target.cpu().detach().numpy()
    current_output = output.cpu().detach().numpy()
    
    return np.mean((current_target-current_output)**2,axis=0)

def calculate_pmae(target, output):

    current_target = target.cpu().detach().numpy()
    current_output = output.cpu().detach().numpy()
    
    return np.mean(np.abs(current_target-current_output),axis=0)
