"""
Plotting functions for ReLU neurons
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import PLOTLY_TEMPLATE
from neuron_generators import generate_random_neurons


def relu_np(x):
    """ReLU activation function"""
    return np.maximum(0, x)


def plot_relu_neuron_sum(num_neurons, x_range, seed, show_individual):
    """
    Plot the sum of ReLU neurons
    
    Args:
        num_neurons: Number of neurons to visualize
        x_range: Range of x-axis
        seed: Random seed for reproducibility
        show_individual: Whether to show individual neurons
    """
    weights, biases, alphas = generate_random_neurons(num_neurons, x_range, seed)
    x = np.linspace(-x_range, x_range, 2000)

    individual_outputs = []
    for i in range(num_neurons):
        neuron_output = alphas[i] * relu_np(weights[i] * x + biases[i])
        individual_outputs.append(neuron_output)

    total_output = np.sum(individual_outputs, axis=0)

    fig = go.Figure()

    if show_individual and num_neurons <= 10:
        palette = px.colors.qualitative.Plotly
        colors = [palette[i % len(palette)] for i in range(num_neurons)]
        for i, (neuron_out, color) in enumerate(zip(individual_outputs, colors)):
            fig.add_trace(go.Scatter(
                x=x, y=neuron_out,
                mode='lines',
                name=f'ReLU {i+1}: α={alphas[i]:.2f}',
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.6
            ))

    fig.add_trace(go.Scatter(
        x=x, y=total_output,
        mode='lines',
        name='TOTAL: f(x) = Σ αᵢ·ReLU(wᵢx+bᵢ)',
        line=dict(color='purple', width=4)
    ))

    y_min, y_max = total_output.min(), total_output.max()
    y_margin = max(abs(y_min), abs(y_max)) * 0.2 if max(abs(y_min), abs(y_max)) > 0 else 1

    fig.update_layout(
        title=f'⚡ ReLU Neuron Network: Sum of {num_neurons} Neurons',
        xaxis_title='x',
        yaxis_title='f(x)',
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        xaxis=dict(range=[-x_range, x_range]),
        yaxis=dict(range=[y_min - y_margin, y_max + y_margin]),
        height=600
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig
