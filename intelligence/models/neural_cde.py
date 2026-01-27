import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CDEFunc(nn.Module):
    """
    The vector field f(z) that drives the CDE:
    dZ_t = f(Z_t) dX_t
    
    f maps latent state Z (batch, hidden_dim) -> Matrix (batch, hidden_dim, input_dim)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # A simple MLP to parameterize the vector field
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim * input_dim)
        self.act = nn.SiLU() # Euclidean-safe activation
        
    def forward(self, z):
        # z: (batch, hidden_dim)
        h = self.linear1(z)
        h = self.act(h)
        # Output: (batch, hidden * input)
        out = self.linear2(h)
        
        # STABILIZATION: Tanh activation on the vector field bounds the rate of change.
        # This prevents the hidden state from exploding over long sequences (256 steps).
        out = torch.tanh(out) 
        
        # Reshape to (batch, hidden, input) for matrix multiplication
        return out.view(z.size(0), self.hidden_dim, self.input_dim)

class NeuralCDE(nn.Module):
    """
    A Neural Controlled Differential Equation (CDE) model.
    It treats the input sequence X as a continuous control path and solves:
    Z_T = Z_0 + int_0^T f(Z_t) dX_t
    
    Uses explicit Euler solver on the observation grid (simplest implementation).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initial state encoder (maps first observation to z0)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # The vector field f(z)
        self.func = CDEFunc(input_dim, hidden_dim)
        
        # Decoder / Readout (optional, if used as standalone)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            last_hidden: (batch, hidden_dim) - The final state of the CDE
        """
        batch_size, seq_len, _ = x.shape
        
        # Z_0 depends on X_0
        z = self.encoder(x[:, 0, :])
        z = F.softplus(z) # Ensure positive or active start? Optional.
        
        # Integration loop (Explicit Euler)
        # dZ = f(Z) * dX
        # Z_{t+1} = Z_t + f(Z_t) @ (X_{t+1} - X_t)
        
        # Precompute dX (control increments)
        # dX shape: (batch, seq_len-1, input_dim)
        dX = x[:, 1:, :] - x[:, :-1, :]
        
        for t in range(seq_len - 1):
            # 1. Evaluate vector field
            # term: (batch, hidden, input)
            mat_field = self.func(z)
            
            # 2. Get control increment
            # dx_t: (batch, input_dim) -> (batch, input_dim, 1)
            dx_t = dX[:, t, :].unsqueeze(-1)
            
            # 3. Matrix-Vector Multiplication: f(z) @ dx
            # (batch, hidden, input) @ (batch, input, 1) -> (batch, hidden, 1)
            dz = torch.bmm(mat_field, dx_t).squeeze(-1)
            
            # 4. Update
            z = z + dz
            
            # Optional: Intermediate regularizations or simplified gating could go here
            if self.dropout.p > 0:
                z = self.dropout(z)
                
        # Z is now Z_T (final state)
        return z

    def get_sequence(self, x):
        """
        Returns full sequence of hidden states for troubleshooting/attention mechanisms.
        """
        batch_size, seq_len, _ = x.shape
        z = self.encoder(x[:, 0, :])
        dX = x[:, 1:, :] - x[:, :-1, :]
        
        states = [z]
        for t in range(seq_len - 1):
            mat_field = self.func(z)
            dx_t = dX[:, t, :].unsqueeze(-1)
            dz = torch.bmm(mat_field, dx_t).squeeze(-1)
            z = z + dz
            states.append(z)
            
        return torch.stack(states, dim=1) # (batch, seq_len, hidden_dim)
