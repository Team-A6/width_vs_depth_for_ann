"""
Activation functions for neural network visualizations
"""

import numpy as np
import torch


def sigmoid_np(u):
    """NumPy sigmoid function"""
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))


def sigmoid_t(u):
    """PyTorch sigmoid function"""
    return torch.sigmoid(u)


def relu_np(u):
    """NumPy ReLU function"""
    return np.maximum(0, u)


def bump_np(x, c, w, k):
    """
    NumPy bump function
    
    Args:
        x: Input array
        c: Center position
        w: Width
        k: Sharpness parameter
    """
    a = c - w / 2.0
    b = c + w / 2.0
    return sigmoid_np(k * (x - a)) - sigmoid_np(k * (x - b))


def bump_t(x, c, w, k=10.0):
    """
    PyTorch bump function
    
    Args:
        x: Input tensor
        c: Center position
        w: Width
        k: Sharpness parameter
    """
    return sigmoid_t(k * (x - (c - w/2.0))) - sigmoid_t(k * (x - (c + w/2.0)))
