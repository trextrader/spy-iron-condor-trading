import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalDiffusionHead(nn.Module):
    """
    Conditional Diffusion Model for Time Series Forecasting.
    
    Refines a latent trajectory using a denoising process conditioned on the 
    Mamba backbone's embedding.
    
    Architecture:
    - Epsilon-Predictor: Residual MLP taking (x_t, t, condition)
    - Schedule: Linear Beta Schedule
    """
    def __init__(
        self, 
        input_dim: int = 4,    # e.g., r, rho, d, v (4 features)
        cond_dim: int = 512,   # Mamba d_model
        hidden_dim: int = 256,
        horizon: int = 32, 
        n_steps: int = 100     # Diffusion steps (keep small for speed)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.n_steps = n_steps
        
        # Time Embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition Projection (Mamba -> Hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Denoising Network (Residual MLP)
        # Input: Flat trajectory (horizon * input_dim) to keep correlations
        flat_dim = horizon * input_dim
        
        self.net = nn.Sequential(
            nn.Linear(flat_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, flat_dim)
        )
        
        # Diffusion Constants (no parameters)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, axis=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    def forward(self, x_start, condition, t=None):
        """
        Training Forward Pass:
        1. Sample random t
        2. Add noise to x_start -> x_t
        3. Predict noise epsilon
        
        Args:
            x_start: Ground truth trajectory (B, H, F)
            condition: Mamba embedding (B, D)
            t: Optional time steps (B,). If None, sampled uniformly.
            
        Returns:
            loss: MSE(noise, predicted_noise)
        """
        B, H, n_feats = x_start.shape
        device = x_start.device
        
        # 1. Sample t
        if t is None:
            t = torch.randint(0, self.n_steps, (B,), device=device).long()
            
        # 2. Add Noise
        noise = torch.randn_like(x_start)
        
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        
        # 3. Predict Noise
        # Flatten x_t for MLP: (B, H*F)
        x_flat = x_t.view(B, -1)
        
        # Conditioning
        c_emb = self.cond_proj(condition) # (B, Hidden)
        t_emb = self.time_mlp(t)          # (B, Hidden)
        
        # Concatenate: [Noisy_Input, Condition, Time]
        inp = torch.cat([x_flat, c_emb, t_emb], dim=-1)
        
        pred_noise_flat = self.net(inp)
        pred_noise = pred_noise_flat.view(B, H, n_feats)
        
        # return pred_noise
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, condition, n_samples=1):
        """
        Generate trajectory via reverse diffusion.
        
        Args:
            condition: Mamba embedding (B, D)
            n_samples: Number of samples per condition (default 1)
            
        Returns:
            x_final: Generated trajectory (B, H, F)
        """
        B = condition.shape[0]
        device = condition.device
        
        # Start from pure noise
        x = torch.randn((B, self.horizon, self.input_dim), device=device)
        
        # Reverse Loop
        for i in reversed(range(self.n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Predict noise
            x_flat = x.view(B, -1)
            c_emb = self.cond_proj(condition)
            t_emb = self.time_mlp(t)
            inp = torch.cat([x_flat, c_emb, t_emb], dim=-1)
            
            eps_theta = self.net(inp).view(B, self.horizon, self.input_dim)
            
            # Update x_{t-1}
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            
            # Mean
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps_theta)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
                
        return x
