import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

from model import UNet256, Diffusion

def train_diffusion(dataloader, device, cfg):
    """
    Train a DDPM‐style diffusion model using U-Net and cosine noise schedule.

    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        Dataloader yielding real training images.
    device : torch.device
        Device on which to run training ('cuda' or 'cpu').
    cfg : dict
        Configuration dictionary containing model parameters, optimizer settings, and save paths.

    Returns:
    --------
    loss_vals : list
        Loss per epoch.

    """
    # Setup output directories
    save_root = cfg['save']['save_dir']
    ckpt_folder = os.path.join(save_root, cfg['save']['checkpoints_folder'])
    sample_folder = os.path.join(save_root, cfg['training']['sample_dir'])
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(sample_folder, exist_ok=True)

    # Create the diffusion model and helper
    diff_cfg = cfg['diffusion']
    diffusion = Diffusion(T=diff_cfg['T'], device=device)
    model = UNet256(
        time_dim=cfg['model']['time_dim'],
        base_ch=cfg['model']['base_ch']
    ).to(device)

    # Optimizer setup
    optim = torch.optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'])

    loss_vals = []

    # Training loop
    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{cfg['training']['epochs']}",
            leave=False
        )

        for real in pbar:
            real = real.to(device)
            B = real.size(0)

            # Sample random timestep per image in batch
            t = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)

            # Compute noise prediction loss
            loss = diffusion.p_losses(model, real, t)

            # Backward and optimize
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

        avg_loss = epoch_loss / len(dataloader)
        loss_vals.append(avg_loss)

        # Save loss history and checkpoint
        np.save(os.path.join(ckpt_folder, "loss.npy"), np.array(loss_vals))
        torch.save(model.state_dict(), os.path.join(ckpt_folder, f"model_epoch{epoch:03d}.pth"))

        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        # Plot and save loss curve
        epochs = list(range(1, len(loss_vals) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_vals, label='Average Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Diffusion Model Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_folder, "Epoch_Loss_Diffusion.png"),
                    dpi=300, bbox_inches='tight', pad_inches=0)

        # Generate samples every few epochs
        if epoch % cfg['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                n_samples = min(64, len(dataloader.dataset))
                samples = diffusion.sample(model, n=n_samples)

            # Save sample grid
            grid = vutils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
            vutils.save_image(grid, os.path.join(sample_folder, f"epoch{epoch:03d}.png"))

    return loss_vals