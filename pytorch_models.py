"""
PyTorch model definitions and training functions
"""

import torch
import torch.nn as nn


def get_activation(activation_name):
    """Select activation function"""
    if activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "tanh":
        return nn.Tanh()
    else:
        return nn.ReLU()


class DynamicMLP(nn.Module):
    """Dynamic MLP model"""
    def __init__(self, input_dim, hidden_layers, neurons_per_layer, output_dim, activation):
        super(DynamicMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(get_activation(activation))
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(get_activation(activation))
        
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def generate_synthetic_data(n_samples):
    """Generate synthetic regression data - Complex sine/cosine combination"""
    torch.manual_seed(42)
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    
    # COMPLEX FUNCTION
    y = (torch.sin(2*x) + 
         0.5*torch.cos(5*x) + 
         0.3*torch.sin(3*x) + 
         0.2*torch.cos(7*x))
    
    # Slight noise
    y = y + 0.1*torch.randn_like(x)
    
    return x, y


def train_single_model(model, x_train, y_train, epochs, lr, batch_size):
    """Train a single model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
    
    return loss_history
