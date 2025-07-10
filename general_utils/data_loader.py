from torch.utils.data import Dataset
from torchvision import transforms
import pydicom, torch
import os
import glob


class RSNADataset(Dataset):
    """
    Custom PyTorch Dataset for loading RSNA chest X-ray images stored in .dcm format.

    This dataset reads DICOM files from a specified root directory, applies optional
    transformations, and returns normalized images as PyTorch tensors.

    Parameters:
    -----------
    root : str
        Path to the directory containing .dcm files.
    transform : torchvision.transforms.Compose, optional
        A set of transformations to apply to the image. If None, a default
        transformation pipeline is used.

    Attributes:
    -----------
    paths : list
        List of full file paths to .dcm files.
    transform : torchvision.transforms.Compose
        Composed transformation pipeline to apply to each image.
    """

    def __init__(self, root, transform=None):
        # Collect all .dcm file paths from the given root directory
        self.paths = sorted(glob.glob(os.path.join(root, "*.dcm")))

        # Define default transformations if none are provided
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),             # Convert numpy array to PIL Image
            transforms.Resize(256),              # Resize shorter side to 256
            transforms.CenterCrop(256),          # Crop center 256x256 region
            transforms.ToTensor(),               # Convert to torch.Tensor and scale to [0, 1]
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        """Return the total number of .dcm files."""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Load and return a transformed image from the dataset.

        Parameters:
        -----------
        idx : int
            Index of the image to retrieve.

        Returns:
        --------
        torch.Tensor
            Transformed image tensor of shape [1, 256, 256] with values in [-1, 1].
        """
        path = self.paths[idx]
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype("float32")

        # DICOM images are grayscale; add channel dimension before transforming
        img = self.transform(img[..., None])
        return img