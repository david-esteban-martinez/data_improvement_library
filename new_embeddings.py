"""
Image Embedding Generation Module

This module processes images and generates feature embeddings using a pretrained deep learning model.
It loads image files, applies transformations, and extracts embeddings in batches to optimize performance.

Key Features:
- Loads images from a specified directory and applies preprocessing.
- Uses a Vision Transformer (ViT) model for feature extraction.
- Supports batch processing for efficiency.
- Saves extracted embeddings in a pickle file for later use.

Functions:
- `create_new_embeddings(directory, output_dir, output_name, embedding_model, batch_size)`:
  Processes images, extracts embeddings, and saves them.

Example Usage:
    create_new_embeddings(
        directory="ALL/",
        output_dir="embeddings/",
        output_name="img_vectors.pkl",
        embedding_model=None,
        batch_size=32
    )

Dependencies:
- os, pickle, numpy, timm, PIL, torch, tqdm
"""

import os
import pickle

import numpy as np
import timm
from PIL import Image
import torch
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm


def create_new_embeddings(directory, output_dir, output_name, embedding_model, batch_size):
    """
    Main function to generate embeddings for images using a deep learning model.

    Args:
        directory (str): Path to the directory containing image files.
        output_dir (str): Path where the processed embeddings will be stored.
        output_name (str): Name of the output file.
        embedding_model (torch.nn.Module or None): Pretrained model for embedding extraction.
         If None, a default model is used.
        batch_size (int): Number of images to process in each batch.
    """
    image_data = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the list of image files sorted numerically based on filename
    file_list = sorted([file for file in os.listdir(directory)], key=lambda x: int(x.split(".")[0]))

    # Load or initialize the embedding model
    if embedding_model is not None:
        model = embedding_model
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
    else:
        model = timm.create_model("timm/vit_large_patch14_clip_336.openai", pretrained=True).to(device)
        model.eval()
        model.classifier = torch.nn.Identity()

        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

    # Process images in batches
    for i in tqdm(range(0, len(file_list), batch_size)):
        batch_files = file_list[i:i + batch_size]

        # Load images and convert to RGB
        img_batch = [Image.open(os.path.join(directory, filename)).convert("RGB") for filename in batch_files]

        # Apply transformation to images
        batch_tensors = [transform(img) for img in img_batch]

        # Convert batch to a single tensor and move to GPU if available
        batch_tensors = torch.stack(batch_tensors).to(device)

        # Perform batch inference to extract embeddings
        with torch.no_grad():
            batch_out = model(batch_tensors)

        # Store the extracted embeddings
        image_data.append(batch_out.cpu().detach().numpy())

    # Convert list of arrays into a single numpy array
    image_data = np.vstack(image_data)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the generated embeddings as a pickle file
    with open(os.path.join(output_dir, output_name), "wb") as f:
        pickle.dump(image_data, f)

    print(f"Embeddings saved to {os.path.join(output_dir, output_name)}")


if __name__ == "__main__":
    # Example usage
    create_new_embeddings(directory="ALL/", output_dir="embeddings/", output_name="img_vectors.pkl",
                          embedding_model=None, batch_size=32)
