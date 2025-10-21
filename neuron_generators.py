"""
Generators for random neuron parameters
"""

import numpy as np


def generate_random_neurons(num_neurons, x_range, seed):
    """
    Generate random sigmoid neuron parameters
    
    Args:
        num_neurons: Number of neurons
        x_range: Range of x values
        seed: Random seed
        
    Returns:
        weights, biases, alphas
    """
    np.random.seed(seed)
    weights = np.random.uniform(-3, 3, num_neurons)
    biases = np.random.uniform(-x_range*0.8, x_range*0.8, num_neurons)
    alphas = np.random.uniform(-2, 2, num_neurons)
    return weights, biases, alphas


def generate_random_bump_neurons(num_bumps, x_range, seed):
    """
    Generate random bump neuron parameters
    
    Args:
        num_bumps: Number of bump neurons
        x_range: Range of x values
        seed: Random seed
        
    Returns:
        centers, widths, sharpnesses, alphas
    """
    np.random.seed(seed)
    centers = np.random.uniform(-x_range*0.8, x_range*0.8, num_bumps)
    widths = np.random.uniform(0.5, 4.0, num_bumps)
    sharpnesses = np.random.uniform(2.0, 15.0, num_bumps)
    alphas = np.random.uniform(-2.0, 2.0, num_bumps)
    return centers, widths, sharpnesses, alphas
