"""
Plotting functions for sigmoid neurons
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import PLOTLY_TEMPLATE
from activation_functions import sigmoid_np
from neuron_generators import generate_random_neurons


def plot_neuron_sum(num_neurons, x_range, seed, show_individual):
    """
    Plot the sum of sigmoid neurons
    
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
        neuron_output = alphas[i] * sigmoid_np(weights[i] * x + biases[i])
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
                name=f'Neuron {i+1}: α={alphas[i]:.2f}',
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.6
            ))

    fig.add_trace(go.Scatter(
        x=x, y=total_output,
        mode='lines',
        name='TOTAL: f(x) = Σ αᵢ·σ(wᵢx+bᵢ)',
        line=dict(color='blue', width=4)
    ))

    y_min, y_max = total_output.min(), total_output.max()
    y_margin = max(abs(y_min), abs(y_max)) * 0.2 if max(abs(y_min), abs(y_max)) > 0 else 1

    fig.update_layout(
        title=f'🧠 Sigmoid Neuron Network: Sum of {num_neurons} Neurons',
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


def plot_three_graphs(c, w, k, alpha, x_range):
    """
    Show left sigmoid, right sigmoid, and bump difference
    
    Args:
        c: Center position
        w: Width
        k: Sharpness
        alpha: Amplitude
        x_range: Range of x-axis
    """
    x = np.linspace(-x_range, x_range, 2000)
    a = c - w / 2.0
    b = c + w / 2.0
    
    y_left = sigmoid_np(k * (x - a))
    y_right = sigmoid_np(k * (x - b))
    y_bump = alpha * (y_left - y_right)
    
    # 3 SUBPLOTS
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('🔴 LEFT SIGMOID: σ(k(x - a))',
                       '🟢 RIGHT SIGMOID: σ(k(x - b))',
                       '🔵 BUMP FUNCTION: B(x) = α · [Left - Right]'),
        vertical_spacing=0.1
    )
    
    # LEFT SIGMOID
    fig.add_trace(go.Scatter(x=x, y=y_left, mode='lines', name='σ(k(x-a))',
                            line=dict(color='red', width=3)), row=1, col=1)
    fig.add_vline(x=a, line_dash="dash", line_color="darkred", opacity=0.7, row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=1, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
    
    # RIGHT SIGMOID
    fig.add_trace(go.Scatter(x=x, y=y_right, mode='lines', name='σ(k(x-b))',
                            line=dict(color='green', width=3)), row=2, col=1)
    fig.add_vline(x=b, line_dash="dash", line_color="darkgreen", opacity=0.7, row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=2, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    
    # BUMP
    fig.add_trace(go.Scatter(x=x, y=y_bump, mode='lines', name='Bump',
                            line=dict(color='blue', width=3.5)), row=3, col=1)
    fig.add_vline(x=a, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_vline(x=b, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)
    fig.add_vline(x=c, line_dash="dash", line_color="orange", opacity=0.8, row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=3, col=1)
    
    # Layout
    fig.update_xaxes(title_text="x", range=[-x_range, x_range], row=3, col=1)
    fig.update_yaxes(title_text="σ(k(x-a))", range=[-0.1, 1.2], row=1, col=1)
    fig.update_yaxes(title_text="σ(k(x-b))", range=[-0.1, 1.2], row=2, col=1)
    fig.update_yaxes(title_text="B(x)", row=3, col=1)
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        template=PLOTLY_TEMPLATE,
        title_text="Sigmoid Difference Analysis: 3 Stages"
    )
    
    return fig
