import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def visualize_data(batch: torch.Tensor, save_dir: str):
    """
    Visualizes and saves a grid of the first 16 images from a batch of images.

    Parameters:
    -----------
    batch : torch.Tensor
        A batch of images with shape [B, C, H, W]. Assumes images are in the range [-1, 1].

    save_dir : str
        Directory where the image grid will be saved.

    Saves:
    ------
    An image grid of the first 16 images as 'image_grid_first_16.png' in `save_dir`.
    """
    # Display tensor metadata
    print("Batch shape:", batch.shape)
    print("Dtype:", batch.dtype, "Min/Max:", batch.min().item(), batch.max().item())

    # Create a grid of the first 16 images, normalize to [0,1] for visualization
    grid = vutils.make_grid(batch[:16], nrow=4, normalize=True, value_range=(-1, 1))

    # Plot and save the image grid
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, "image_grid_first_16.png"),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close()