"""
Plotting functions for bump neurons
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOTLY_TEMPLATE, QUAL_PLOTLY
from activation_functions import bump_np
from neuron_generators import generate_random_bump_neurons


def plot_single_bump(center, width, sharpness, amplitude, x_range):
    """
    Show single bump function in detail
    
    Args:
        center: Center position
        width: Width of bump
        sharpness: Sharpness parameter
        amplitude: Amplitude
        x_range: Range of x-axis
    """
    x = np.linspace(-x_range, x_range, 2000)
    y = amplitude * bump_np(x, center, width, sharpness)
    
    a = center - width / 2.0
    b = center + width / 2.0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name=f'Bump(c={center:.1f}, w={width:.1f})',
        line=dict(color='blue', width=2.5)
    ))
    
    # Reference lines
    fig.add_vline(x=center, line_dash="dot", line_color="orange", opacity=0.7,
                  annotation_text=f"c={center:.1f}")
    fig.add_vline(x=a, line_dash="dot", line_color="red", opacity=0.5)
    fig.add_vline(x=b, line_dash="dot", line_color="green", opacity=0.5)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    
    y_margin = abs(amplitude) * 0.2 if amplitude != 0 else 1
    
    fig.update_layout(
        title='🎯 Bump Function: B(x) = α · [σ(k(x-a)) - σ(k(x-b))]',
        xaxis_title='x',
        yaxis_title='B(x; c, w, k)',
        template=PLOTLY_TEMPLATE,
        xaxis=dict(range=[-x_range, x_range]),
        yaxis=dict(range=[min(0, amplitude) - y_margin, max(0, amplitude) + y_margin]),
        height=500
    )
    
    return fig


def plot_bump_sum(num_bumps, x_range, seed, show_individual):
    """
    Plot the sum of bump neurons
    
    Args:
        num_bumps: Number of bump neurons
        x_range: Range of x-axis
        seed: Random seed
        show_individual: Whether to show individual bumps
    """
    centers, widths, sharpnesses, alphas = generate_random_bump_neurons(num_bumps, x_range, seed)
    x = np.linspace(-x_range, x_range, 2000)

    individual_outputs = []
    for i in range(num_bumps):
        bump_output = alphas[i] * bump_np(x, centers[i], widths[i], sharpnesses[i])
        individual_outputs.append(bump_output)

    total_output = np.sum(individual_outputs, axis=0)

    fig = go.Figure()

    # Individual bumps
    if show_individual and num_bumps <= 10:
        colors = [QUAL_PLOTLY[i % len(QUAL_PLOTLY)] for i in range(num_bumps)]
        for i, (bump_out, color) in enumerate(zip(individual_outputs, colors)):
            fig.add_trace(go.Scatter(
                x=x, y=bump_out,
                mode='lines',
                name=f'Bump {i+1}: α={alphas[i]:.2f}',
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.6
            ))
            fig.add_vline(x=centers[i], line_dash="dot", line_color=color, opacity=0.3)

    # Total
    fig.add_trace(go.Scatter(
        x=x, y=total_output,
        mode='lines',
        name='TOTAL: f(x) = Σ αᵢ·B(x; cᵢ, wᵢ, kᵢ)',
        line=dict(color='blue', width=4)
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    y_min, y_max = total_output.min(), total_output.max()
    y_margin = max(abs(y_min), abs(y_max)) * 0.2 if max(abs(y_min), abs(y_max)) > 0 else 1

    fig.update_layout(
        title=f'🎯 Bump Neuron Network: Sum of {num_bumps} Local Functions',
        xaxis_title='x',
        yaxis_title='f(x)',
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        xaxis=dict(range=[-x_range, x_range]),
        yaxis=dict(range=[y_min - y_margin, y_max + y_margin]),
        height=600
    )

    return fig
