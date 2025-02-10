# data_improvement_library
A group of techniques and methods which can potentlially improve visual explainability models based on the Learning to Rank aproach
# Usage Guide

Each module in the library requires a set of input variables that define its functionality. This section describes these variables and provides usage examples for each module.

## Data Augmentation Module

### Input Variables

- `data_dir` (str): Path to the input data file containing user interactions.
- `vector_dir` (str): Path to the file containing image embeddings.
- `image_dir` (str): Path to the directory where images are stored.
- `output_dir` (str): Path where the processed data will be stored.
- `output_name` (str, optional): Name of the output file. Default: `TRAIN_IMG`.
- `embedding_model` (`torch.nn.Module`, optional): Pretrained model for embedding extraction. If left empty, a pretrained ViT Large 14 model is used by default.
- `no_aug` (bool, optional): If `True`, disables data augmentation.
- `batch_size` (int, optional): Number of images processed per batch. Default: 32.
- `apply_all` (bool, optional): If `True`, applies all available transformations.
- `labels` (list, optional): List of column names in the dataset.

### Output Variables

This module does not produce a return value. Instead, it saves the processed data in `pickle` format based on the input variables.

### Usage Example

```python
from data_augmentation import augment_data

# Process images with data augmentation and embedding extraction
augment_data(data_dir="data/user_data.pkl", vector_dir="data/image_vectors.pkl",
     image_dir="images/", output_dir="processed_data/",
     output_name="TRAIN_IMG", embedding_model=None,
     no_aug=False, batch_size=32, apply_all=True, labels=None)
```

## Embedding Generation Module (`new_embeddings.py`)

### Input Variables

- `directory` (str): Directory where images are stored.
- `output_dir` (str): Directory where generated embeddings will be saved.
- `output_name` (str): Name of the output file.
- `embedding_model` (`torch.nn.Module` or `None`): Pretrained model for embedding extraction. If left empty, a pretrained ViT Large 14 model is used by default.
- `batch_size` (int): Number of images processed per batch.

### Output Variables

This module does not return a value. Instead, it saves the processed data in `pickle` format.

### Usage Example

```python
from new_embeddings import create_new_embeddings

# Generate embeddings for a set of images
create_new_embeddings(directory="images/", output_dir="embeddings/", output_name="img_vectors.pkl",
     embedding_model=None, batch_size=32)
```

## Negative Selection with PU Learning (`pu_negatives.py`)

### Input Variables

- `data_file` (str): Path to the user interaction data file.
- `vector_file` (str): Path to the image embeddings file.
- `outdir_name` (str): Output directory where the processed dataset will be saved.
- `centroid` (int, optional): Centroid percentile (default: 90).
- `factor` (float, optional): Distance adjustment factor.
- `labels` (list, optional): Column labels in the dataset.

### Output Variables

This module does not return a value. Instead, it saves the processed data in `pickle` format.

### Usage Example

```python
from pu_negatives import resample_negatives

# Generate balanced negative samples
resample_negatives(data_file="data/user_data.pkl", vector_file="data/image_vectors.pkl",
     outdir_name="processed_data/", centroid=90, factor=1.0)
```

## Positive Selection with PU Learning (`pu_positives.py`)

This module can be used in two main ways:

1. **As a Callback** in model training with `PyTorch Lightning`, allowing dynamic updates of user centroids in each epoch.
2. **As Preprocessing**, applied before training to the dataset.

### Usage as Callback

The module provides the `CallbackEndPositives` class, which extends `Callback` from `PyTorch Lightning`. Its purpose is to update user centroids and resample positive samples at the end of each training epoch.

For proper functionality, the model must meet the following requirements:

- **Dataset in `train_dataloader` must contain:**
  - `datamodule.image_embeddings`: Tensor of image embeddings.
  - `dataframe`: Must include columns `id_user`, `id_img`, `id_restaurant`, `take`.
- **Embedding access:** Must be indexable by `id_img` in the `dataframe`.
- **Embedding format:** Must be a tensor convertible to NumPy (`.cpu().detach().numpy()`).
- **Data sampling:** Must allow modification of `pu_dataset` in the dataset.
- **DataModule requirements:** Must provide `image_embeddings` and handle `pu_dataset` correctly.

#### Usage Example

```python
from pu_positives import CallbackEndPositives
from pytorch_lightning import Trainer

# Create an instance of the callback
callback_end = CallbackEndPositives()

# Configure the trainer with the callback
trainer = Trainer(callbacks=[callback_end])

# Start training
trainer.fit(model)
```

### Usage as Preprocessing

This module can also generate a dataset with new positive samples selected based on user similarity before training. This is achieved by calculating user centroids and resampling positive samples. The functionality is divided into two functions: `centroid_users` for computing user centroids and `resample_positives` for updating the dataset with new positive samples using cosine similarity filtering.

#### Input Variables

For the function `centroid_users`:

- `dataframe` (pandas.DataFrame): Dataset of user-image pairs.
- `vectors` (numpy.array): Image embedding representations.
- `labels` (list, optional): List of column names in the dataset.

For the function `resample_positives`:

- `dataframe` (pandas.DataFrame): Dataset of user-image pairs.
- `centroids` (dict): Dictionary where keys represent user IDs, and values are user image centroids as embeddings.
- `k` (int): Number of similar users considered for adding positive samples per user.

#### Output Variables

- `centroid_users` returns `centroids`, a dictionary where keys are user IDs and values are user image centroids as embeddings.
- `resample_positives` returns the updated `dataframe` with new positive samples.

#### Usage Example

```python
from pu_positives import centroid_users, resample_positives
import pandas as pd
import numpy as np

# Load the dataset
dataframe = pd.read_pickle("data/user_data.pkl")

# Load image embeddings
image_vectors = np.load("data/image_vectors.npy")

# Compute user centroids
centroids = centroid_users(dataframe, image_vectors)

# Generate a new dataset with balanced positive samples
new_dataframe = resample_positives(dataframe, centroids, 3)

# Save the processed dataset
new_dataframe.to_pickle("processed_data/balanced_dataset.pkl")
```

This README provides a comprehensive guide on using the modules in this library. Each section explains the required inputs, expected outputs, and provides working examples to facilitate integration into your workflow.

