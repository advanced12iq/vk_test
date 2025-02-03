#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader
from src import (
    process_split,
    config,
    LogoDataset,
    SiameseNetwork,
    train_siamese_network
)

def main():
    parser = argparse.ArgumentParser(description='Train Siamese Network')
    parser.add_argument('--train-annotations', required=True,
                      help='Path to training annotations JSON')
    parser.add_argument('--val-annotations', required=True,
                      help='Path to validation annotations JSON')
    parser.add_argument('--output-prefix', default='osld',
                      help='Output file prefix for processed data')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Input batch size')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    args = parser.parse_args()

    # Preprocess data
    print("Preprocessing data...")
    process_split(args.train_annotations, 
                 config.PRODUCT_IMAGES_DIR,
                 config.LOGO_IMAGES_DIR,
                 config.TRAIN_OUTPUT_PREFIX)
    
    process_split(args.val_annotations,
                 config.PRODUCT_IMAGES_DIR,
                 config.LOGO_IMAGES_DIR,
                 config.VAL_OUTPUT_PREFIX)

    # Create datasets
    train_dataset = LogoDataset(
        product_crops_path=config.TRAIN_OUTPUT_PREFIX + "_product_crops.npy",
        logo_images_path=config.TRAIN_OUTPUT_PREFIX + "_logo_images.npy",
        transform_product = config.TRANSFORM_PRODUCT,
        transform_logo = config.TRANSFORM_LOGO,
    )
    
    val_dataset = LogoDataset(
        product_crops_path=config.VAL_OUTPUT_PREFIX + "_product_crops.npy",
        logo_images_path=config.VAL_OUTPUT_PREFIX + "_logo_images.npy",
        transform_product = config.TRANSFORM_PRODUCT,
        transform_logo = config.TRANSFORM_LOGO,
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = SiameseNetwork()

    # Train
    train_siamese_network(
        model,
        train_loader,
        device=config.DEVICE,
    )

    #Save Model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("Model saved to", config.MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()  