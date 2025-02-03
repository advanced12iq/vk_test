#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader
import datetime
import os
from src import (
    process_split,
    config,
    LogoDataset,
    SiameseNetwork,
    train_siamese_network
)

def main():
    # Preprocess data
    print("Preprocessing data...")
    process_split(config.TRAIN_ANNOTATIONS_PATH, 
                 config.PRODUCT_IMAGES_DIR,
                 config.LOGO_IMAGES_DIR,
                 config.TRAIN_OUTPUT_PREFIX)
    
    process_split(config.VAL_ANNOTATIONS_PATH,
                 config.PRODUCT_IMAGES_DIR,
                 config.LOGO_IMAGES_DIR,
                 config.VAL_OUTPUT_PREFIX)

    # Create datasets
    train_dataset = LogoDataset(
        product_crops_path=config.TRAIN_OUTPUT_PREFIX + "_product_crops.npy",
        logo_images_path=config.TRAIN_OUTPUT_PREFIX + "_logo_images.npy",
        transform = config.TRANSFORM_PRODUCT
    )
    
    val_dataset = LogoDataset(
        product_crops_path=config.VAL_OUTPUT_PREFIX + "_product_crops.npy",
        logo_images_path=config.VAL_OUTPUT_PREFIX + "_logo_images.npy",
        transform = config.TRANSFORM_PRODUCT
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model
    model = SiameseNetwork()

    # Train
    train_siamese_network(
        model,
        train_loader,
        val_loader,
        device=config.DEVICE,
    )

    #Save Model
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, f'siamese_model{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'))
    print("Model saved to", config.MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()  