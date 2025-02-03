# config.py

import os
import torch
from torchvision import transforms
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')


DATASET_DIR = os.path.join(RAW_DATA_DIR, 'osld')
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'annotations')
TRAIN_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, 'osld-train.json')
VAL_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, 'osld-val.json')
TEST_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, 'osld-test.json')

PRODUCT_IMAGES_DIR = os.path.join(DATASET_DIR, 'product-images')
LOGO_IMAGES_DIR = os.path.join(DATASET_DIR, 'logo-images')


TRAIN_OUTPUT_PREFIX = os.path.join(PROCESSED_DATA_DIR, 'train')
VAL_OUTPUT_PREFIX = os.path.join(PROCESSED_DATA_DIR, 'val')
TEST_OUTPUT_PREFIX = os.path.join(PROCESSED_DATA_DIR, 'test')

RESIZE_SIZE = (128, 128)

BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
MARGIN = 1.0

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', f'siamese_model{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRANSFORM_PRODUCT = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORM_LOGO = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])