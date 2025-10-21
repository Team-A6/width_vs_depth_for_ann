"""
Training visualization functions
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOTLY_TEMPLATE
from pytorch_models import DynamicMLP, generate_synthetic_data, train_single_model


def train_models(epochs, batch_size, n_samples,
                 m1_layers, m1_neurons, m1_lr, m1_activation,
                 m2_layers, m2_neurons, m2_lr, m2_activation,
                 m3_layers, m3_neurons, m3_lr, m3_activation,
                 m4_layers, m4_neurons, m4_lr, m4_activation):
    """Train 4 models sequentially and visualize results"""
    
    x_train, y_train = generate_synthetic_data(n_samples)
    
    # Test data
    x_test = torch.linspace(-4, 4, 500).unsqueeze(1)
    y_test = (torch.sin(2*x_test) + 
              0.5*torch.cos(5*x_test) + 
              0.3*torch.sin(3*x_test) + 
              0.2*torch.cos(7*x_test))
    
    models_config = [
        {"name": "Model 1: Wide-Shallow", "layers": int(m1_layers), "neurons": int(m1_neurons), 
         "lr": m1_lr, "activation": m1_activation, "color": "#E74C3C", "dash": "solid"},
        {"name": "Model 2: Medium-Medium", "layers": int(m2_layers), "neurons": int(m2_neurons),
         "lr": m2_lr, "activation": m2_activation, "color": "#27AE60", "dash": "dash"},
        {"name": "Model 3: Narrow-Deep", "layers": int(m3_layers), "neurons": int(m3_neurons),
         "lr": m3_lr, "activation": m3_activation, "color": "#3498DB", "dash": "dashdot"},
        {"name": "Model 4: Very Narrow-Very Deep", "layers": int(m4_layers), "neurons": int(m4_neurons),
         "lr": m4_lr, "activation": m4_activation, "color": "#9B59B6", "dash": "dot"}
    ]
    
    # 2 SUBPLOTS: Loss (top) + Approximation (bottom)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.4, 0.6],
        subplot_titles=('📉 Training Loss Comparison',
                       '🎯 Complex Sin/Cos Function Approximation'),
        vertical_spacing=0.12
    )
    
    all_losses = []
    trained_models = []
    
    for idx, config in enumerate(models_config):
        model = DynamicMLP(1, config['layers'], config['neurons'], 1, config['activation'])
        total_params = sum(p.numel() for p in model.parameters())
        
        loss_history = train_single_model(model, x_train, y_train, int(epochs), config['lr'], int(batch_size))
        
        all_losses.append(loss_history)
        trained_models.append(model)
        
        # LOSS GRAPH (top)
        fig.add_trace(go.Scatter(
            x=list(range(len(loss_history))),
            y=loss_history,
            mode='lines',
            name=f"{config['name']} ({total_params}p)",
            line=dict(color=config['color'], width=3, dash=config['dash']),
            legendgroup=f'model{idx}',
            showlegend=True
        ), row=1, col=1)
        
        # FUNCTION APPROXIMATION (bottom)
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test).detach().numpy()
        
        fig.add_trace(go.Scatter(
            x=x_test.numpy().flatten(),
            y=y_pred.flatten(),
            mode='lines',
            name=f"{config['name']} (Loss: {loss_history[-1]:.4f})",
            line=dict(color=config['color'], width=3.5, dash=config['dash']),
            legendgroup=f'model{idx}',
            showlegend=False
        ), row=2, col=1)
    
    # True function (bottom graph)
    fig.add_trace(go.Scatter(
        x=x_test.numpy().flatten(),
        y=y_test.numpy().flatten(),
        mode='lines',
        name='🎯 True Function',
        line=dict(color='black', width=4.5),
        showlegend=True
    ), row=2, col=1)
    
    # Training data (bottom graph)
    fig.add_trace(go.Scatter(
        x=x_train.numpy().flatten(),
        y=y_train.numpy().flatten(),
        mode='markers',
        name='📍 Training Data',
        marker=dict(color='gray', size=5, opacity=0.4),
        showlegend=True
    ), row=2, col=1)
    
    # Training region highlight
    fig.add_vrect(x0=-3, x1=3, fillcolor="yellow", opacity=0.1, layer="below",
                  annotation_text="Training Region", annotation_position="top left",
                  row=2, col=1)
    
    # Layout
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss (MSE)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="x", range=[-4, 4], row=2, col=1)
    fig.update_yaxes(title_text="y", row=2, col=1)
    
    fig.update_layout(
        height=900,
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        title_text="🔥 Width vs Depth: Complex Sin/Cos Function Learning",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Status message
    best_idx = int(np.argmin([l[-1] for l in all_losses]))
    final_status = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    final_status += "✅ ALL MODELS TRAINED SUCCESSFULLY!\n\n"
    
    final_status += "🎯 TARGET FUNCTION:\n"
    final_status += "   f(x) = sin(2x) + 0.5·cos(5x) + 0.3·sin(3x) + 0.2·cos(7x)\n"
    final_status += "   • 4 different frequencies (2, 5, 3, 7)\n"
    final_status += "   • Mixed sine and cosine\n\n"
    
    final_status += f"🏆 BEST MODEL:\n"
    final_status += f"   → {models_config[best_idx]['name']}\n"
    final_status += f"   → Final Loss: {all_losses[best_idx][-1]:.6f}\n"
    final_status += f"   → Parameters: {sum(p.numel() for p in trained_models[best_idx].parameters())}\n\n"
    
    final_status += "📊 ALL MODELS:\n"
    for idx, config in enumerate(models_config):
        params = sum(p.numel() for p in trained_models[idx].parameters())
        final_status += f"\n{config['name']}:\n"
        final_status += f"   • Architecture: {config['layers']}L × {config['neurons']}N = {params}p\n"
        final_status += f"   • Activation: {config['activation'].upper()}\n"
        final_status += f"   • Final Loss: {all_losses[idx][-1]:.6f}\n"
    
    final_status += "\n💡 INTERACTIVE FEATURES:\n"
    final_status += "   • Zoom in on the graph\n"
    final_status += "   • Click legend models to show/hide\n"
    final_status += "   • Hover for detailed values\n"
    final_status += "   • Compare models in the bottom graph!\n"
    
    return fig, final_status
        
        
