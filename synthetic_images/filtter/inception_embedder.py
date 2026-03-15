import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3


def build_inception_embedder(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()
    model = inception_v3(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model, preprocess, device


def load_real_paths_by_class(train_dir, train_labels_path=None):
    train_dir = Path(train_dir)
    train_labels_path = Path(train_labels_path) if train_labels_path else train_dir / "dataset.json"

    with open(train_labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)["labels"]

    real_paths_by_class = {}
    for filename, class_index in labels:
        real_paths_by_class.setdefault(int(class_index), []).append(train_dir / filename)
    return real_paths_by_class


def _load_batch(image_paths, preprocess):
    images = []
    valid_paths = []
    for image_path in image_paths:
        image_path = Path(image_path)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        images.append(preprocess(image))
        valid_paths.append(image_path)

    if not images:
        return None, []
    return torch.stack(images, dim=0), valid_paths


def extract_embeddings(image_paths, model, preprocess, device, batch_size=32):
    embeddings = []
    ordered_paths = []

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_tensor, valid_paths = _load_batch(batch_paths, preprocess)
            if batch_tensor is None:
                continue

            batch_tensor = batch_tensor.to(device)
            batch_embeddings = model(batch_tensor).detach().cpu().numpy()
            embeddings.append(batch_embeddings)
            ordered_paths.extend(valid_paths)

    if not embeddings:
        return np.empty((0, 2048), dtype=np.float32), []
    return np.concatenate(embeddings, axis=0), ordered_paths