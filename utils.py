import os
import numpy as np
from spectral import envi
from PIL import Image
import random

def get_random_kiwi_image():
    """Get a random kiwi image from the showcase directory."""
    kiwi_dir = "data/showcase/kiwi"
    files = [f for f in os.listdir(kiwi_dir) if f.endswith('.bin')]
    if not files:
        raise FileNotFoundError("No kiwi images found in showcase directory")
    return os.path.join(kiwi_dir, random.choice(files))

def load_spectral_data(bin_file):
    """Load spectral data from .bin file."""
    hdr_file = bin_file.replace('.bin', '.hdr')
    if not os.path.exists(hdr_file):
        raise FileNotFoundError(f"Header file not found: {hdr_file}")
    
    img = envi.open(hdr_file, bin_file)
    return img

def spectral_to_rgb(spectral_data):
    """Convert spectral data to approximate RGB values."""
    # This is a simplified conversion 
    if len(spectral_data.shape) == 3:
        # Assuming the data is in format (height, width, bands)
        # Simple RGB approximation using different bands
        r = np.mean(spectral_data[:, :, 0:5], axis=2)  # Red approximation
        g = np.mean(spectral_data[:, :, 5:10], axis=2)  # Green approximation
        b = np.mean(spectral_data[:, :, 10:15], axis=2)  # Blue approximation
        
        # Normalize to 0-255 range
        rgb = np.stack([r, g, b], axis=2)
        rgb = ((rgb - rgb.min()) * (255.0 / (rgb.max() - rgb.min()))).astype(np.uint8)
        return rgb
    return None

def get_average_rgb(rgb_image):
    """Calculate average RGB values from an image."""
    if rgb_image is None:
        return None
    return np.mean(rgb_image, axis=(0, 1)).astype(int) 