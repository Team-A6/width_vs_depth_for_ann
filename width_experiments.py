"""
Width experiment plotting functions
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOTLY_TEMPLATE
from activation_functions import sigmoid_np, relu_np, bump_t


def plot_sigmoid_width_experiment(width_values_str, seed, x_range):
    """Width experiment with sigmoid neurons"""
    try:
        width_values = [int(w.strip()) for w in width_values_str.split(',') if w.strip()]
        if not width_values:
            width_values = [2, 4, 8, 16]
    except Exception:
        width_values = [2, 4, 8, 16]
    
    np.random.seed(seed)
    x = np.linspace(-x_range, x_range, 2000)
    
    # SUBPLOTS
    fig = make_subplots(
        rows=len(width_values), cols=1,
        subplot_titles=[f'🧠 Sigmoid Neuron: W = {W}' for W in width_values],
        vertical_spacing=0.08
    )
    
    for idx, W in enumerate(width_values, 1):
        if W <= 0:
            continue
        
        weights = np.random.uniform(-3, 3, W)
        biases = np.random.uniform(-x_range*0.8, x_range*0.8, W)
        alphas = np.random.uniform(-2, 2, W)
        
        y = np.zeros_like(x)
        for i in range(W):
            y += alphas[i] * sigmoid_np(weights[i] * x + biases[i])
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'W={W}',
            line=dict(color='blue', width=3)
        ), row=idx, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=idx, col=1)
        fig.update_xaxes(title_text="x" if idx == len(width_values) else "", row=idx, col=1)
        fig.update_yaxes(title_text="f(x)", row=idx, col=1)
    
    fig.update_layout(
        height=400*len(width_values),
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        title_text="🧠 Sigmoid Neuron Width Experiment: Global Activation"
    )
    
    return fig


def plot_relu_width_experiment(width_values_str, seed, x_range):
    """Width experiment with ReLU neurons"""
    try:
        width_values = [int(w.strip()) for w in width_values_str.split(',') if w.strip()]
        if not width_values:
            width_values = [2, 4, 8, 16]
    except Exception:
        width_values = [2, 4, 8, 16]
    
    np.random.seed(seed)
    x = np.linspace(-x_range, x_range, 2000)
    
    # SUBPLOTS
    fig = make_subplots(
        rows=len(width_values), cols=1,
        subplot_titles=[f'⚡ ReLU Neuron: W = {W}' for W in width_values],
        vertical_spacing=0.08
    )
    
    for idx, W in enumerate(width_values, 1):
        if W <= 0:
            continue
        
        weights = np.random.uniform(-3, 3, W)
        biases = np.random.uniform(-x_range*0.8, x_range*0.8, W)
        alphas = np.random.uniform(-2, 2, W)
        
        y = np.zeros_like(x)
        for i in range(W):
            y += alphas[i] * relu_np(weights[i] * x + biases[i])
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'W={W}',
            line=dict(color='green', width=3)
        ), row=idx, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=idx, col=1)
        fig.update_xaxes(title_text="x" if idx == len(width_values) else "", row=idx, col=1)
        fig.update_yaxes(title_text="f(x)", row=idx, col=1)
    
    fig.update_layout(
        height=400*len(width_values),
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        title_text="⚡ ReLU Neuron Width Experiment: Piecewise Linear Activation"
    )
    
    return fig


def plot_bump_width_experiment(width_values_str, seed, x_range):
    """Width experiment with bump neurons"""
    try:
        width_values = [int(w.strip()) for w in width_values_str.split(',') if w.strip()]
        if not width_values:
            width_values = [2, 4, 8, 16]
    except Exception:
        width_values = [2, 4, 8, 16]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.linspace(-x_range, x_range, 2000)

    # SUBPLOTS
    fig = make_subplots(
        rows=len(width_values), cols=1,
        subplot_titles=[f'🎯 Bump Neuron: W = {W}' for W in width_values],
        vertical_spacing=0.08
    )
    
    for idx, W in enumerate(width_values, 1):
        if W <= 0:
            continue
            
        centers = torch.linspace(-x_range * 0.8, x_range * 0.8, W)
        widths = torch.full((W,), 6.0 / (2.0 * W))
        # sign vector (+/-1)
        signs = torch.where(torch.rand(W) > 0.5, torch.ones(W), -torch.ones(W))
        alphas = (0.5 + torch.rand(W)) * signs
        
        y = torch.zeros_like(x)
        for ci, wi, ai in zip(centers, widths, alphas):
            y = y + ai * bump_t(x, ci, wi, k=10.0)
        
        fig.add_trace(go.Scatter(
            x=x.detach().numpy(), y=y.detach().numpy(),
            mode='lines',
            name=f'W={W}',
            line=dict(color='red', width=3)
        ), row=idx, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=idx, col=1)
        fig.update_xaxes(title_text="x" if idx == len(width_values) else "", row=idx, col=1)
        fig.update_yaxes(title_text="f(x)", row=idx, col=1)
    
    fig.update_layout(
        height=400*len(width_values),
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        title_text="🎯 Bump Neuron Width Experiment: Local Activation"
    )
    
    return fig
