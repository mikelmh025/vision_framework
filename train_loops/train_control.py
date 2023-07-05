import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import loss_factory
import cvxpy as cp
import time

# from train_loops.default import train_epoch_default
# from train_loops.drops import train_epoch_drops
# from train_loops.peer import train_epoch_peer

def train_control(train_package, loop_type):
    if loop_type == 'default':
        if 'train_epoch_default' not in globals(): from train_loops.default import train_epoch_default
        return train_epoch_default(train_package)
    elif loop_type == 'peer':
        if 'train_epoch_peer' not in globals(): from train_loops.peer import train_epoch_peer
        return train_epoch_peer(train_package)
    elif loop_type == 'drops':
        if 'train_epoch_drops' not in globals(): from train_loops.drops import train_epoch_drops
        return train_epoch_drops(train_package)



