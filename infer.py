#!/usr/bin/env python3
import argparse
import torch
import os
from src import (
    config,
    SiameseNetwork,
    predict_logo,
)


def main():
    parser = argparse.ArgumentParser(description='Logo Detection Inference')
    parser.add_argument('--product', required=True,
                      help='Path to product crop image')
    parser.add_argument('--logos-dir', required=True,
                      help='Directory containing logo images')
    parser.add_argument('--checkpoint', required=True,
                      help='Model checkpoint path')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Similarity threshold')
    args = parser.parse_args()

    # Load model
    model = torch.load(os.path.join(config.MODEL_SAVE_PATH, args.checkpoint), weights_only=False, map_location=config.DEVICE)
    model.eval()

    # Run prediction
    result = predict_logo(
        model,
        args.product,
        args.logos_dir,
        threshold=args.threshold,
        device=config.DEVICE
    )

    # Print results
    print(f"Threshold: {result['threshold']}")
    print("Similarities:")
    for logo_name, similarity in result["similarities"].items():
       print(f"  {logo_name}: {similarity:.4f}")
    print("----------------------------------------")
    print(f"Max similarity: {result['max_similarity']:.4f}")
    print(f"Contains logo: {'YES' if result['detected'] else 'NO'}")


if __name__ == '__main__':
    main()