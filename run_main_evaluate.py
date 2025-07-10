import torch
import yaml
import os
import argparse
from torch.utils.data import DataLoader

from general_utils.data_loader import RSNADataset
from src.wgan.evaluate import evaluate_wgan_stats
from src.diffusion.evaluate import evaluate_diffusion_stats
from general_utils.plotting import visualize_data


def main(config_path: str):
    """
        Main entry point for evaluating GAN or diffusion models using a provided YAML config.

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
    # Load Test Dataset & DataLoader
    # ------------------------
    ds_cfg = cfg['dataset']
    dataset = RSNADataset(ds_cfg['test_root'])
    dataloader = DataLoader(
        dataset,
        batch_size=ds_cfg['batch_size'],
        shuffle=ds_cfg['shuffle'],
        num_workers=ds_cfg['num_workers'],
        pin_memory=ds_cfg['pin_memory']
    )
    print("--- Test Dataset Loaded ---")

    # ------------------------
    # Model Evaluation
    # ------------------------
    model_name = cfg['model']['name']
    if model_name == 'wgan':
        print("--- Evaluating WGAN GP model ---")
        fake_loader = evaluate_wgan_stats(
            dataloader_real=dataloader,
            device=device,
            cfg=cfg
        )
        visualize_data(next(iter(fake_loader)), os.path.join(cfg['save']['save_dir'], cfg['save']['checkpoints_folder'], cfg['output']['results_folder']))
    elif model_name == 'diffusion':
        print("--- Evaluating Diffusion model ---")
        loss_vals = evaluate_diffusion_stats(
            dataloader_real=dataloader,
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
    main(args.config)