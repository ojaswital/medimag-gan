import torch
import yaml
import os
import argparse
from torch.utils.data import DataLoader

from general_utils.data_loader import RSNADataset
from src.wgan.train import train_wgan_gp
from src.diffusion.train import train_diffusion
from general_utils.plotting import visualize_data


def main(config_path: str):
    """
    Main entry point for training GAN or diffusion models using a provided YAML config.

    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file.
    """
    # ------------------------
    # Load Configuration
    # ------------------------
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # ------------------------
    # Select Device
    # ------------------------
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    # ------------------------
    # Load Dataset & DataLoader
    # ------------------------
    ds_cfg = cfg['dataset']
    dataset = RSNADataset(ds_cfg['root'])
    dataloader = DataLoader(
        dataset,
        batch_size=ds_cfg['batch_size'],
        shuffle=ds_cfg['shuffle'],
        num_workers=ds_cfg['num_workers'],
        pin_memory=ds_cfg['pin_memory']
    )

    # ------------------------
    # Visualize Sample Batch
    # ------------------------
    save_dir = os.path.join(cfg['save']['save_dir'], cfg['save']['checkpoints_folder'])
    visualize_data(next(iter(dataloader)), save_dir)
    print("--- Train Dataset Loaded ---")

    # ------------------------
    # Model Training
    # ------------------------
    model_name = cfg['model']['name']
    if model_name == 'wgan':
        # Train Wasserstein GAN with gradient penalty
        print("--- Training WGAN GP model ---")
        lossD_vals, lossG_vals = train_wgan_gp(
            dataloader=dataloader,
            device=device,
            cfg=cfg
        )
    elif model_name == 'diffusion':
        # Train denoising diffusion probabilistic model
        print("--- Training Diffusion model ---")
        loss_vals = train_diffusion(
            dataloader=dataloader,
            device=device,
            cfg=cfg
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'wgan' or 'diffusion'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train WGAN-GP via config file")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config_gan.yaml',
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()
    main(args.config, args.model)