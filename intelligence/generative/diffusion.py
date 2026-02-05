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
    Neural CDE backbone's latent state.
    
    Architecture:
    - Epsilon-Predictor: Residual MLP taking (x_t, t, condition)
    - Schedule: Linear Beta Schedule
    """
    def __init__(
        self, 
        input_dim: int = 4,    # e.g., r, rho, d, v (4 features)
        cond_dim: int = 512,   # Neural CDE hidden state dim
        hidden_dim: int = 256,
        horizon: int = 32, 
        n_steps: int = 100,     # Diffusion steps (keep small for speed)
        cond_drop_prob: float = 0.2 # V42: For classifier-free guidance
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.n_steps = n_steps
        self.cond_drop_prob = cond_drop_prob
        
        # Time Embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition Projection (Neural CDE -> Hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Denoising Network (Residual MLP)
        flat_dim = horizon * input_dim
        
        # V42: FiLM modulation parameters (Scale and Shift)
        self.film_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, flat_dim * 2) # produces [scale, shift]
        )
        
        self.net = nn.Sequential(
            nn.Linear(flat_dim + hidden_dim, hidden_dim * 2), # x_t + time
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
            condition: Neural CDE latent state (B, D)
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
        
        # Conditioning (V42: Conditioning Dropout for CFG)
        if self.training and self.cond_drop_prob > 0:
            mask = (torch.rand(B, 1, device=device) > self.cond_drop_prob).float()
            condition = condition * mask
            
        # V42: FiLM Bridge (Conditioning scale/shift)
        c_emb = self.cond_proj(condition) # (B, Hidden)
        film_params = self.film_net(c_emb) # (B, 2 * flat_dim)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Apply FiLM to noisy input: x' = scale * x + shift
        x_modulated = x_flat * (1 + gamma) + beta
        
        t_emb = self.time_mlp(t)          # (B, Hidden)
        
        # Concatenate: [Modulated_Input, Time]
        inp = torch.cat([x_modulated, t_emb], dim=-1)
        
        pred_noise_flat = self.net(inp)
        pred_noise = pred_noise_flat.view(B, H, n_feats)
        
        # return pred_noise
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, condition, n_samples=1, guidance_scale=2.0):
        """
        Generate trajectory via reverse diffusion using Classifier-Free Guidance (CFG).
        
        Args:
            condition: CDE latent state (B, D)
            n_samples: Number of samples per condition
            guidance_scale: w factor for CFG (amplifies conditioning). 1.0 = no guidance.
            
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
            
            # V42: Classifier-Free Guidance step
            # 1. Prediction with conditioning
            x_flat = x.view(B, -1)
            c_emb = self.cond_proj(condition)
            film_cond = self.film_net(c_emb)
            gamma_c, beta_c = film_cond.chunk(2, dim=-1)
            x_cond = x_flat * (1 + gamma_c) + beta_c
            t_emb = self.time_mlp(t)
            eps_cond = self.net(torch.cat([x_cond, t_emb], dim=-1)).view(B, self.horizon, self.input_dim)
            
            # 2. Prediction without conditioning (null-mask)
            null_cond = torch.zeros_like(condition)
            c_emb_null = self.cond_proj(null_cond)
            film_null = self.film_net(c_emb_null)
            gamma_n, beta_n = film_null.chunk(2, dim=-1)
            x_null = x_flat * (1 + gamma_n) + beta_n
            eps_null = self.net(torch.cat([x_null, t_emb], dim=-1)).view(B, self.horizon, self.input_dim)
            
            # 3. CFG combination: eps = eps_null + w * (eps_cond - eps_null)
            eps_theta = eps_null + guidance_scale * (eps_cond - eps_null)
            
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
