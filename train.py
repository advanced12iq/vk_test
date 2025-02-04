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
    
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--n-epochs',type=int, default=config.NUM_EPOCHS,
                      help='Number of training epochs')
    parser.add_argument('--margin', type=float, default=config.MARGIN,
                      help='Constrative loss margin')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                      help='Learning rate ')
    args = parser.parse_args()

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = SiameseNetwork()

    # Train
    train_siamese_network(
        model,
        train_loader,
        val_loader,
        lr=args.lr,
        n_epochs=args.n_epochs,
        margin=args.margin,
        batch_size=args.batch_size,
        device=config.DEVICE,
    )

    #Save Model
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, f'siamese_model{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'))
    print("Model saved to", config.MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()  