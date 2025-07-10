import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.utils import spectral_norm
import torchvision

class Generator256(nn.Module):
    """
    Generator model for 256x256 image synthesis using transposed convolutions.

    Parameters:
    -----------
    z_dim : int
        Dimension of the latent vector (input noise).
    ngf : int
        Base number of generator filters.

    Architecture:
    -------------
    Transposed convolutional layers progressively upscale the latent vector
    from 1x1 → 256x256 through intermediate resolutions.
    """

    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),  # 1x1 → 4x4
            nn.BatchNorm2d(ngf*8), nn.ReLU(),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),  # 4x4 → 8x8
            nn.BatchNorm2d(ngf*4), nn.ReLU(),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),  # 8x8 → 16x16
            nn.BatchNorm2d(ngf*2), nn.ReLU(),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),    # 16x16 → 32x32
            nn.BatchNorm2d(ngf), nn.ReLU(),

            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False),   # 32x32 → 64x64
            nn.BatchNorm2d(ngf//2), nn.ReLU(),

            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1, bias=False),  # 64x64 → 128x128
            nn.BatchNorm2d(ngf//4), nn.ReLU(),

            nn.ConvTranspose2d(ngf//4, 1, 4, 2, 1, bias=False),     # 128x128 → 256x256
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        """
        Generate image from input noise z.

        Parameters:
        -----------
        z : torch.Tensor
            Latent noise tensor of shape [B, z_dim, 1, 1].

        Returns:
        --------
        torch.Tensor
            Generated image of shape [B, 1, 256, 256] in [-1, 1].
        """
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminator model for 256x256 grayscale images using spectral norm and no sigmoid.

    Parameters:
    -----------
    ndf : int
        Base number of discriminator filters.

    Returns:
    --------
    torch.Tensor
        Real-valued scores per image sample (no sigmoid).
    """

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(1, ndf, 4, 2, 1, bias=False)),  # 256 → 128
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),  # 128 → 64
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),  # 64 → 32
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),  # 32 → 16
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)),  # 16 → 13 (1 channel)
        )

    def forward(self, x):
        """
        Compute raw realism score for each input image.

        Parameters:
        -----------
        x : torch.Tensor
            Input image tensor of shape [B, 1, 256, 256].

        Returns:
        --------
        torch.Tensor
            Flattened tensor of shape [B] with raw discriminator scores.
        """
        x = self.net(x)                    # [B, 1, H, W]
        x = F.adaptive_avg_pool2d(x, 1)    # [B, 1, 1, 1]
        return x.view(-1)                  # Flatten to [B]


def gradient_penalty(D, real, fake, λ=10.0):
    """
    Compute the gradient penalty for WGAN-GP.

    Parameters:
    -----------
    D : nn.Module
        The discriminator model.
    real : torch.Tensor
        Batch of real images [B, C, H, W].
    fake : torch.Tensor
        Batch of fake images [B, C, H, W].
    λ : float
        Regularization strength (default: 10.0).

    Returns:
    --------
    torch.Tensor
        Scalar gradient penalty loss.
    """
    B, C, H, W = real.shape
    α = torch.rand(B, 1, 1, 1, device=real.device)
    interp = (α * real + (1 - α) * fake).requires_grad_(True)
    d_interp = D(interp)

    grads = grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad_norm = grads.view(B, -1).norm(2, dim=1)
    penalty = λ * ((grad_norm - 1) ** 2).mean()
    return penalty
