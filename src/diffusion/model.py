import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

# -----------------------------------------------------------------------------
# Noise schedule (cosine) & helper functions for Diffusion model
# -----------------------------------------------------------------------------
def cosine_beta_schedule(T, s=0.008):
    """
    Create a cosine schedule for noise variances (betas) over T timesteps.
    s: small offset to avoid singularities at endpoints.
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)  # timesteps 0…T
    # Compute cumulative product of alphas via cosine curve
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to 1 at t=0
    # Betas = fraction of noise added at each step
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clamp to avoid extreme values
    return torch.clamp(betas, 0.0001, 0.9999)

class Diffusion:
    """
    Implements forward diffusion (q_sample) and reverse sampling (sample) of DDPM.
    """
    def __init__(self, T=1000, device='cuda'):
        self.device = device
        self.T = T
        # Precompute betas and derived alphas
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1 - betas
        # Cumulative product of alphas for direct use in formulas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion: sample noisy image x_t from clean x0 at timestep t.
        x_t = sqrt(alpha_cumprod[t]) * x0 + sqrt(1 - alpha_cumprod[t]) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self.sqrt_alphas_cumprod[t].view(-1,1,1,1) * x0 +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1) * noise
        )

    def p_losses(self, denoise_model, x0, t):
        """
        Compute the MSE loss between the true noise and the model's predicted noise.
        """
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)   # noisy image at step t
        pred_noise = denoise_model(xt, t)  # model predicts noise
        return F.mse_loss(pred_noise, noise)  # MSE loss

    @torch.no_grad()
    def sample(self, denoise_model, n, z_dim=None):
        """
        Reverse diffusion process: start from random noise x_T and iteratively denoise to x_0.
        """
        x = torch.randn(n, 1, 256, 256, device=self.device)  # start with Gaussian noise
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = denoise_model(x, t)               # predict noise residual
            alpha_t = self.alphas_cumprod[i]
            alpha_prev = self.alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0, device=self.device)
            beta_t = 1 - alpha_t
            # DDPM reverse update
            coef1 = 1 / torch.sqrt(1 - beta_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_t)
            x = coef1 * (x - coef2 * eps)
            if i > 0:
                # add random noise except at final step
                noise = torch.randn_like(x)
                sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * beta_t)
                x += sigma * noise
        return x.clamp(-1,1)  # clamp output to [-1,1]

# -----------------------------------------------------------------------------
# U-Net backbone for noise prediction (denoiser)
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two Conv2d→GroupNorm→SiLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """Downsample block: MaxPool2d → DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """Upsample block: ConvTranspose2d → concat skip → DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # skip connection
        return self.conv(x)

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding + small MLP.
    Embeds discrete timestep t into a dense vector for conditioning.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim*4),
        )
    def forward(self, t):
        half = self.dim // 2
        # sinusoidal embedding
        emb = torch.sin(torch.arange(half, device=t.device) * t.unsqueeze(-1)
                       / (10000 ** (2 * torch.arange(half, device=t.device)/self.dim)))
        emb = torch.cat([emb, emb.cos()], dim=-1)
        return self.lin(emb)

class UNet256(nn.Module):
    """
    U‐Net with a single time‐conditioning at the bottleneck.
    - Encoder downsamples from 1→64→128→256→512 channels.
    - The bottleneck adds the time embedding to the 512‐channel tensor.
    - Decoder upsamples 512→256→128→64 channels, concatenating skips.
    """
    def __init__(self, time_dim=128, base_ch=64):
        super().__init__()
        # 1) Time‐step embedder (outputs time_dim*4 = 512 dim)
        self.time_mlp = TimeEmbedding(time_dim)

        # 2) Encoder (downsampling path)
        self.inc   = DoubleConv(1,         base_ch)       # 1→64
        self.down1 = Down(base_ch,        base_ch*2)     # 64→128
        self.down2 = Down(base_ch*2,      base_ch*4)     # 128→256
        self.down3 = Down(base_ch*4,      base_ch*8)     # 256→512

        # 3) Bottleneck
        self.bot   = DoubleConv(base_ch*8, base_ch*8)     # 512→512

        # 4) Decoder (upsampling path) – note the channel math:
        #    Up(in_ch, out_ch) does:
        #      x = ConvTranspose2d(in_ch→out_ch)
        #      x = cat([x, skip], dim=1)          # out_ch + skip_ch
        #      x = DoubleConv(in_ch=out_ch+skip_ch → out_ch)
        #
        #   For up3: in_ch   = 512  (b output channels)
        #           out_ch  = 256  (we want to go 512→256)
        #   skip x3 has 256 channels, so
        #   after up: tensor has 256, cat skip → 512, conv(512→256) matches.
        self.up3 = Up(base_ch*8,  base_ch*4)  # 512→256
        self.up2 = Up(base_ch*4,  base_ch*2)  # 256→128
        self.up1 = Up(base_ch*2,  base_ch)    # 128→64

        # 5) Final 1×1 convolution back to 1 channel
        self.outc = nn.Conv2d(base_ch, 1, 1)  # 64→1

    def forward(self, x, t):
        # Embed the timestep (-> [B,512])
        t_emb = self.time_mlp(t)

        # --- Encoder ---
        x1 = self.inc(x)       # [B,  64,256,256]
        x2 = self.down1(x1)    # [B, 128,128,128]
        x3 = self.down2(x2)    # [B, 256, 64, 64]
        x4 = self.down3(x3)    # [B, 512, 32, 32]

        # --- Bottleneck (add time embedding) ---
        # reshape t_emb to [B,512,1,1] so broadcast adds across H×W
        b = self.bot(x4 + t_emb.view(-1, x4.size(1), 1, 1))  # [B,512,32,32]

        # --- Decoder (no further time‐adds) ---
        # up3: maps 512→256, cat with x3(256)→512 in conv→256
        u3 = self.up3(b, x3)   # [B,256, 64, 64]
        # up2: maps 256→128, cat with x2(128)→256 in conv→128
        u2 = self.up2(u3, x2)  # [B,128,128,128]
        # up1: maps 128→64,  cat with x1( 64)→128 in conv→ 64
        u1 = self.up1(u2, x1)  # [B, 64,256,256]

        # Final conv to produce 1‐channel output
        return self.outc(u1)   # [B, 1,256,256]

