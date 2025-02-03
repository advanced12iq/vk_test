# src/__init__.py

# Import core components from each file
from .dataset import LogoDataset
from .process import process_split
from .models.siamese_model import SiameseNetwork
from .models.loss import contrastive_loss
from .trainer import train_siamese_network
from .predict import predict_logo


# Define public API
__all__ = [
    # Core components
    'LogoDataset',
    'process_split',
    'SiameseNetwork',
    'contrastive_loss',
    'train_siamese_network',
    'test_model',
    'predict_logo',
    'config'
]

# Optional: Package version
__version__ = "0.1.0"