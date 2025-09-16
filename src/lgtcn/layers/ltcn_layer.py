import torch
import torch.nn as nn

class LTCNLayer(nn.Module):
    """Standard Liquid Time Constant Network layer (without graph components)."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Time constant parameters
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        
        # Input transformation
        self.W_in = nn.Linear(in_dim, hidden_dim)
        
        # Recurrent connections
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating mechanism
        self.W_gate = nn.Linear(in_dim + hidden_dim, hidden_dim)
        
        # Bias terms
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor, u: torch.Tensor, dt: float = 5e-2, n_steps: int = 1):
        """Propagate the hidden state through time using LTC dynamics."""
        
        # Expand biases if we are in batch mode
        if x.ndim == 3:
            tau = self.tau.expand(x.size(0), x.size(1), -1)
            bias = self.bias.expand(x.size(0), x.size(1), -1)
        else:
            tau = self.tau.expand(x.size(0), -1)
            bias = self.bias.expand(x.size(0), -1)
        
        for _ in range(n_steps):
            # Input transformation
            u_emb = self.W_in(u)
            
            # Recurrent connections
            rec = self.W_rec(x)
            
            # Gating mechanism
            if x.ndim == 3:
                gate_input = torch.cat([u, x], dim=-1)
            else:
                gate_input = torch.cat([u, x], dim=-1)
            gate = torch.sigmoid(self.W_gate(gate_input))
            
            # LTC dynamics: dx/dt = -x/tau + f(u, x)
            f = torch.tanh(u_emb + rec + bias)
            dx = -x / (tau + 1e-6) + gate * f
            
            x = x + dt * dx
            x = torch.clamp(x, -1.0, 1.0)
            
        return x