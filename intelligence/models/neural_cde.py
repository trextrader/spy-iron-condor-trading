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
        
        # V42: Spectral normalization for global Lipschitz constraint
        self.linear2 = nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim * input_dim))
        
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
        
        # V42: Recalibrated z0 encoder with LN + Tanh
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # The vector field f(z)
        self.func = CDEFunc(input_dim, hidden_dim)
        
        # Decoder / Readout (optional, if used as standalone)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, timestep_mask: torch.Tensor = None, bar_index: torch.Tensor = None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            timestep_mask: Binary mask (batch, seq_len-1) - 1 for new bar, 0 for same bar.
            bar_index: Bar grouping index (batch, seq_len). 
        Returns:
            last_hidden: (batch, hidden_dim) - The final state of the CDE
        """
        batch_size, seq_len, _ = x.shape

        if bar_index is not None:
            # BCS GUARDRAIL
            with torch.no_grad():
                diff = bar_index[:, 1:] - bar_index[:, :-1]
                if not (diff >= 0).all():
                    raise ValueError("BCS Violation: NeuralCDE batch must be grouped by bar_index.")
                if timestep_mask is None:
                    timestep_mask = (diff > 0.5).float()
        
        # V42: Recalibrated initial state
        z = self.encoder(x[:, 0, :])
        
        # Integration loop (V42: Heun / RK2 Solver)
        # 1. Proposal: z_tilde = z_k + f(z_k) @ (dX_k)
        # 2. Correction: z_{k+1} = z_k + 0.5 * (f(z_k) + f(z_tilde)) @ (dX_k)
        
        # Precompute dX (control increments)
        # dX shape: (batch, seq_len-1, input_dim)
        dX = x[:, 1:, :] - x[:, :-1, :]
        
        for t in range(seq_len - 1):
            # BCS: Only advance if this is a new market bar
            # (In training, if 100 rows = 1 bar, mask will be 0 for 99 steps)
            weight = timestep_mask[:, t].unsqueeze(-1) if timestep_mask is not None else 1.0
            
            # 1. Evaluate vector field at current state
            # f_k: (batch, hidden, input)
            f_k = self.func(z)
            
            # 2. Get control increment
            # dx_t: (batch, input_dim) -> (batch, input_dim, 1)
            dx_t = dX[:, t, :].unsqueeze(-1)
            
            # 3. Euler proposal
            dz_euler = torch.bmm(f_k, dx_t).squeeze(-1)
            z_tilde = z + dz_euler
            
            # 4. Evaluate vector field at proposal state
            f_next = self.func(z_tilde)
            
            # 5. Heun update: Trapezoidal correction
            f_avg = 0.5 * (f_k + f_next)
            dz_heun = torch.bmm(f_avg, dx_t).squeeze(-1)
            
            # 6. Final Update with Gating (BCS)
            z = z + weight * dz_heun
            
            # Optional: Intermediate regularizations
            if self.dropout.p > 0 and (weight > 0.5).any() if torch.is_tensor(weight) else self.dropout.p > 0:
                # Only apply dropout on real steps
                z = self.dropout(z)
                
        # Z is now Z_T (final state)
        return z

    def step(self, x_current, x_next, z_prev):
        """
        Stateful incremental update (Phase 5).
        Z_{k+1} = Z_k + int_{t_k}^{t_{k+1}} f(Z_t) dX_t
        
        Args:
            x_current: (batch, input_dim) - Feature at t_k
            x_next: (batch, input_dim) - Feature at t_{k+1}
            z_prev: (batch, hidden_dim) - Hidden state at t_k
            
        Returns:
            z_next: (batch, hidden_dim) - Hidden state at t_{k+1}
        """
        # 1. Proposal (Euler)
        dx = (x_next - x_current).unsqueeze(-1)
        f_k = self.func(z_prev)
        dz_euler = torch.bmm(f_k, dx).squeeze(-1)
        z_tilde = z_prev + dz_euler
        
        # 2. Correction (Heun)
        f_next = self.func(z_tilde)
        f_avg = 0.5 * (f_k + f_next)
        dz_heun = torch.bmm(f_avg, dx).squeeze(-1)
        
        z_next = z_prev + dz_heun
        
        if self.dropout.p > 0:
            z_next = self.dropout(z_next)
            
        return z_next

    def get_sequence(self, x):
        """
        Returns full sequence of hidden states for troubleshooting/attention mechanisms.
        """
        batch_size, seq_len, _ = x.shape
        z = self.encoder(x[:, 0, :])
        dX = x[:, 1:, :] - x[:, :-1, :]
        
        states = [z]
        for t in range(seq_len - 1):
            f_k = self.func(z)
            dx_t = dX[:, t, :].unsqueeze(-1)
            
            dz_euler = torch.bmm(f_k, dx_t).squeeze(-1)
            z_tilde = z + dz_euler
            
            f_next = self.func(z_tilde)
            f_avg = 0.5 * (f_k + f_next)
            dz_heun = torch.bmm(f_avg, dx_t).squeeze(-1)
            
            z = z + dz_heun
            states.append(z)
            
        return torch.stack(states, dim=1) # (batch, seq_len, hidden_dim)
