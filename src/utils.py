import os
import string
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def create_if_noexists(path):
    """Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def next_version(directory, base_name, extension):
    """Generate next available version filename to avoid overwriting.
    
    Finds the next available filename by appending version numbers (v1, v2, etc.)
    to avoid overwriting existing files.
    
    Args:
        directory (str): Directory containing the files.
        base_name (str): Base filename without extension.
        extension (str): File extension including the dot.
        
    Returns:
        str: Complete path with version number that doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    version = 0
    while True:
        path = os.path.join(directory, f"{base_name}-v{version}{extension}")
        if not os.path.exists(path):
            return path
        version += 1


def positions_into_cell(pos, cell):
    """Wrap atomic positions into the unit cell using periodic boundaries.
    
    Maps atomic positions to their equivalent positions within the unit cell
    by applying periodic boundary conditions.
    
    Args:
        pos (torch.Tensor): Atomic positions, shape (n_atoms, 3).
        cell (torch.Tensor): Unit cell matrix, shape (3, 3).
        
    Returns:
        torch.Tensor: Wrapped positions within the unit cell.
    """
    invlat = torch.linalg.inv(cell)
    relpos = pos @ invlat
    relpos = relpos % 1.0
    pos = relpos @ cell
    return pos


def string2slice(s):
    """Convert string representation to Python slice object.
    
    Parses slice notation strings like ':10', '5:', '1:10:2' into
    Python slice objects for array indexing.
    
    Args:
        s (str): String representation of slice (e.g., ':', ':10', '5:15').
        
    Returns:
        slice: Python slice object.
    """
    if ':' in s:
        return slice(*map(lambda x: int(x) if x else None, s.split(':')))
    else:
        return int(s)


class TrainLogger:
    """Logger for training progress and metrics.
    
    Handles logging of training metrics including loss values, learning rates,
    and other training statistics. Provides methods for writing to files and
    computing running averages.
    
    Attributes:
        log_path (str): Path to the log file.
        log_freq (int): Frequency of logging (every N iterations).
        log_split_loss (bool): Whether to log separate position and element losses.
    """
    def __init__(self, file, n_bins, t_min, t_max, log_split_loss=False):
        self.file = file
        self.n_bins = n_bins
        self.t_min = t_min
        self.t_max = t_max
        self.log_split_loss = log_split_loss

        self.reset_loss_bins()

        with open(self.file, 'a') as f:
            #        '      1  0.00010000  0.0010    0.0010 0.0010 0.0010 0.0010 0.0010 0.0010 0.0010 0.0010 0.0010 0.0010'
            f.write('# epoch lr l_tot')
            if self.log_split_loss:
                f.write(' l_pos l_el')
            f.write(f' [binned total loss (x{self.n_bins})]')
            if self.log_split_loss:
                f.write(f' [binned position loss (x{self.n_bins})]')
                f.write(f' [binned element loss (x{self.n_bins})]')
                f.write('')
            f.write('\n')

    def reset_loss_bins(self):
        self.loss_bins_tot = [[] for _ in range(self.n_bins)]
        if self.log_split_loss:
            self.loss_bins_pos = [[] for _ in range(self.n_bins)]
            self.loss_bins_els = [[] for _ in range(self.n_bins)]

    def register_sample_loss(self, t, loss_tot, loss_pos=None, loss_els=None):
        i_loss_bin = int((t - self.t_min) * self.n_bins /
                         (self.t_max - self.t_min))

        self.loss_bins_tot[i_loss_bin].append(loss_tot)
        if self.log_split_loss:
            self.loss_bins_pos[i_loss_bin].append(loss_pos)
            self.loss_bins_els[i_loss_bin].append(loss_els)

    def log(self, epoch, lr):
        loss_tot = np.mean([l for ls in self.loss_bins_tot for l in ls])
        if self.log_split_loss:
            loss_pos = np.mean([l for ls in self.loss_bins_pos for l in ls])
            loss_els = np.mean([l for ls in self.loss_bins_els for l in ls])

        with open(self.file, 'a') as f:
            f.write(f'{epoch:6d} {lr:6.8f} {loss_tot:6.4f}')
            if self.log_split_loss:
                f.write(f' {loss_pos:6.4f} {loss_els:6.4f}')
            f.write(
                '    ' + ' '.join([f'{np.mean(loss_bin):6.4f}' for loss_bin in self.loss_bins_tot]))
            if self.log_split_loss:
                f.write(
                    '    ' + ' '.join([f'{np.mean(loss_bin):6.4f}' for loss_bin in self.loss_bins_pos]))
                f.write(
                    '    ' + ' '.join([f'{np.mean(loss_bin):6.4f}' for loss_bin in self.loss_bins_els]))
            f.write('\n')

        self.reset_loss_bins()


class TensorBoardLogger:
    """TensorBoard logger for visualizing training metrics.
    
    Wraps TensorBoard SummaryWriter to provide convenient logging of scalars,
    images, and other metrics during training and inference.
    
    Attributes:
        writer (SummaryWriter): TensorBoard writer instance.
    """
    def __init__(self, log_dir):
        # Create tensorboard directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar value"""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars under the same main tag"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag, values, step):
        """Log histogram of values"""
        self.writer.add_histogram(tag, values, step)

    def log_model_params(self, model, step):
        """Log histograms of model parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Skip empty tensors
                if param.data.numel() > 0:
                    try:
                        self.log_histogram(f"params/{name}", param.data, step)
                        if param.grad is not None and param.grad.numel() > 0:
                            self.log_histogram(f"grads/{name}", param.grad, step)
                    except ValueError:
                        pass

    def close(self):
        self.writer.close()
