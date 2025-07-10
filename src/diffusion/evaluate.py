import torch
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from scipy.stats import ks_2samp, wasserstein_distance

from model import UNet256, Diffusion


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a float image tensor in the range [-1, 1] to uint8 in the range [0, 255].

    Parameters:
    -----------
    x : torch.Tensor
        Image tensor of shape [B, C, H, W] with values in [-1.0, 1.0].

    Returns:
    --------
    torch.Tensor
        Tensor with dtype uint8 and values in [0, 255].
    """
    return ((x + 1.0) / 2.0 * 255).clamp(0, 255).to(torch.uint8)


def to_rgb299(x_uint8: torch.Tensor) -> torch.Tensor:
    """
    Convert grayscale uint8 image tensor to 3-channel RGB and resize to 299x299 for Inception metrics.

    Parameters:
    -----------
    x_uint8 : torch.Tensor
        Grayscale tensor of shape [B, 1, H, W] and dtype uint8.

    Returns:
    --------
    torch.Tensor
        RGB tensor of shape [B, 3, 299, 299] and dtype uint8.
    """
    # Repeat grayscale channel to create RGB
    x_rgb = x_uint8.repeat(1, 3, 1, 1)  # Shape: [B, 3, H, W]

    # Resize to 299x299 using bilinear interpolation
    return F.interpolate(
        x_rgb.float(), size=(299, 299), mode='bilinear', align_corners=False
    ).to(torch.uint8)


def evaluate_diffusion_stats(dataloader_real, device, cfg):
    """
    Evaluate a trained diffusion model using FID, KID, KS-statistic, and EMD.

    Parameters:
    -----------
    dataloader_real : DataLoader
        DataLoader containing real images.
    device : torch.device
        Computation device ('cuda' or 'cpu').
    cfg : dict
        Configuration dictionary containing paths and model settings.

    Returns:
    --------
    fake_loader : DataLoader
        DataLoader containing generated fake images.
    """

    # Create necessary directories
    ckpt_dir = os.path.join(cfg['save']['save_dir'], cfg['save']['checkpoints_folder'])
    os.makedirs(ckpt_dir, exist_ok=True)
    results_dir = os.path.join(ckpt_dir, cfg['output']['results_folder'])
    os.makedirs(results_dir, exist_ok=True)

    # Setup
    num_real = len(dataloader_real.dataset)
    batch_size = dataloader_real.batch_size
    num_batches = (num_real + batch_size - 1) // batch_size

    # Load trained generator
    model = UNet256(
        time_dim=cfg['model']['time_dim'],
        base_ch=cfg['model']['base_ch']
    ).to(device)
    model_path = os.path.join(ckpt_dir, cfg['output']['best_model_name'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize diffusion helper
    diffusion = Diffusion(T=cfg['diffusion']['T'], device=device)

    # Generate synthetic samples
    fake_imgs = []
    with torch.no_grad():
        for _ in range(num_batches):
            fake = diffusion.sample(model, n=batch_size).cpu()
            fake_imgs.append(fake)
    fake_imgs = torch.cat(fake_imgs, dim=0)[:num_real]  # trim to match
    fake_loader = DataLoader(TensorDataset(fake_imgs), batch_size=batch_size, shuffle=False)

    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(normalize=False).to(device)

    # Update metrics with real images
    for real_batch in dataloader_real:
        real_u8 = to_uint8(real_batch).to(device)
        real_inp = to_rgb299(real_u8)
        fid.update(real_inp, real=True)
        kid.update(real_inp, real=True)

    # Update metrics with fake images
    for fake_batch, in fake_loader:
        fake_u8 = to_uint8(fake_batch).to(device)
        fake_inp = to_rgb299(fake_u8)
        fid.update(fake_inp, real=False)
        kid.update(fake_inp, real=False)

    # Compute metrics
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_score = kid_mean.item()

    # Flatten pixel distributions for KS and EMD
    real_pixels = np.concatenate([batch.numpy().flatten() for batch in dataloader_real.dataset])
    fake_pixels = fake_imgs.numpy().flatten()
    ks_stat, ks_p = ks_2samp(real_pixels, fake_pixels)
    emd = wasserstein_distance(real_pixels, fake_pixels)

    # Display results
    print("--- Diffusion Statistical Evaluation ---")
    print(f"FID Score:            {fid_score:.4f}")
    print(f"KID Score:            {kid_score:.4f} (\u00b1{kid_std.item():.4f})")
    print(f"KS Statistic:         {ks_stat:.4f}, p-value: {ks_p:.4e}")
    print(f"Wasserstein Distance: {emd:.4f}")

    # Save metrics to CSV
    metrics = {
        "FID": fid_score,
        "KID": kid_score,
        "KS-stat": ks_stat,
        "EMD": emd
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print("--- Diffusion Statistical Evaluation metrics saved---")

    # Plot pixel distributions
    plt.figure(figsize=(6, 4))
    plt.hist(real_pixels, bins=50, alpha=0.5, label="Real", density=True)
    plt.hist(fake_pixels, bins=50, alpha=0.5, label="Fake", density=True)
    plt.legend()
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")
    plt.title("Pixel Distribution: Real vs. Fake")
    plt.savefig(os.path.join(results_dir, "Pixel_Distribution.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Q–Q plot comparing quantiles
    plt.figure(figsize=(5, 5))
    stats.probplot(fake_pixels, dist=stats.rv_histogram((np.histogram(real_pixels, bins=100, density=True))), plot=plt)
    plt.title("Q–Q Plot of Fake vs. Real Pixels")
    plt.savefig(os.path.join(results_dir, "Q-Q_Plot.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("--- Diffusion Statistical Evaluation plots saved ---")

    return fake_loader

