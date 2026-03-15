import sys

sys.path.insert(0, r"E:\Pipeline\test_etiquetas\stylegan3")

import glob
import os

import legacy
import numpy as np
import torch
from PIL import Image


def score_images(
    pkl_path: str,
    images_dir: str,
    c_index: int = 0,
    pattern: str = "*.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(pkl_path, "rb") as file:
        data = legacy.load_network_pkl(file)

    if not isinstance(data, dict) or "D" not in data:
        raise ValueError("El .pkl no contiene el discriminador 'D' como dict['D'].")

    discriminator = data["D"].eval().requires_grad_(False).to(device)

    c_dim = getattr(discriminator, "c_dim", 0)
    if c_dim > 0:
        if not (0 <= c_index < c_dim):
            raise ValueError(f"c_index={c_index} fuera de rango. c_dim={c_dim}.")
        c_tensor = torch.zeros([1, c_dim], device=device)
        c_tensor[0, c_index] = 1.0
    else:
        c_tensor = torch.empty([1, 0], device=device)

    paths = sorted(glob.glob(os.path.join(images_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes con patrón {pattern} en {images_dir}")

    def load_img_tensor(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        x = x.unsqueeze(0)
        x = (x / 127.5) - 1.0
        return x.to(device)

    results = []
    with torch.no_grad():
        for path in paths:
            x_tensor = load_img_tensor(path)
            logit = discriminator(x_tensor, c_tensor).item()
            results.append(
                {
                    "path": path,
                    "filename": os.path.basename(path),
                    "logit": float(logit),
                }
            )

    return results