import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from model import Generator256, Discriminator, gradient_penalty

def train_wgan_gp(dataloader, device, cfg):
    """
    Train a WGAN-GP model using a given configuration dictionary.

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
    lossD_vals : list
        Discriminator loss per epoch.
    lossG_vals : list
        Generator loss per epoch.
    """

    # Create checkpoint directory
    ckpt_dir = os.path.join(cfg['save']['save_dir'], cfg['save']['checkpoints_folder'])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Instantiate models
    G = Generator256().to(device)
    D = Discriminator().to(device)

    # Create optimizers
    optG = torch.optim.Adam(G.parameters(), lr=cfg['model']['lr_G'], betas=cfg['model']['betas'])
    optD = torch.optim.Adam(D.parameters(), lr=cfg['model']['lr_D'], betas=cfg['model']['betas'])

    # Track losses for plotting
    lossD_vals, lossG_vals = [], []

    # Training loop
    for epoch in range(1, cfg['model']['num_epochs'] + 1):
        print(f"Epoch {epoch}/{cfg['model']['num_epochs']}")
        epoch_d, epoch_g = 0.0, 0.0

        for real in dataloader:
            real = real.to(device)
            B = real.size(0)

            # Add instance noise to real images
            real_noisy = real + cfg['model']['instance_noise_std'] * torch.randn_like(real)

            # === Train Discriminator ===
            D.zero_grad()
            d_real = D(real_noisy).mean()

            z = torch.randn(B, cfg['model']['z_dim'], 1, 1, device=device)
            fake = G(z)
            fake_noisy = fake.detach() + cfg['model']['instance_noise_std'] * torch.randn_like(fake)
            d_fake = D(fake_noisy).mean()

            gp = gradient_penalty(D, real, fake.detach())
            lossD = d_fake - d_real + gp
            lossD.backward()
            optD.step()
            epoch_d += lossD.item()

            # === Train Generator (multiple steps) ===
            for _ in range(cfg['model']['G_steps_per_D']):
                G.zero_grad()
                z = torch.randn(B, cfg['model']['z_dim'], 1, 1, device=device)
                fake = G(z)
                g_out = D(fake).mean()
                lossG = -g_out
                lossG.backward()
                optG.step()
            epoch_g += lossG.item()

        # Epoch loss averages
        avgD = epoch_d / len(dataloader)
        avgG = epoch_g / len(dataloader)
        lossD_vals.append(avgD)
        lossG_vals.append(avgG)

        # Save losses
        np.save(os.path.join(ckpt_dir, "lossD.npy"), np.array(lossD_vals))
        np.save(os.path.join(ckpt_dir, "lossG.npy"), np.array(lossG_vals))

        # Save model checkpoint
        torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_epoch{epoch}.pth"))

        # Print progress
        print(f"Epoch [{epoch}/{cfg['model']['num_epochs']}]  LossD: {avgD:.4f}  LossG: {avgG:.4f}")

        # Plot loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(lossD_vals) + 1), lossD_vals, label='Discriminator Loss')
        plt.plot(range(1, len(lossG_vals) + 1), lossG_vals, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(ckpt_dir, "Epoch_Loss_Generator_Discriminator.png"),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()

    return lossD_vals, lossG_vals