import os
import json
import random
import shutil
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CLASS_TO_BIN = {
    0: 0, 1: 0, 2: 0, 3: 0, 
    4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
    10: -1
}

def set_seed(seed: int) -> None:
    random.seed(seed)

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS

def load_real_images(train_dir: Path) -> Dict[str, List[Path]]:
    class_to_files = {str(c): [] for c in range(11)}
    json_path = train_dir / "dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No existe {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    labels = data.get("labels", [])
    for item in tqdm(labels, desc="Cargando rutas reales", leave=False):
        img_name, cls_idx = item[0], item[1]
        img_path = train_dir / img_name
        if img_path.exists() and is_image_file(img_path):
            class_to_files[str(cls_idx)].append(img_path)
            
    return {k: v for k, v in class_to_files.items() if v}

def load_synth_images(synth_dir: Path) -> Dict[str, List[Path]]:
    class_to_files = {}
    if not synth_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {synth_dir}")

    folders = [f for f in synth_dir.iterdir() if f.is_dir()]
    for folder in tqdm(folders, desc="Escaneando folders sintéticos", leave=False):
        match = re.match(r"imagenes_(\d+)_ok", folder.name)
        if match:
            cls_str = match.group(1)
            files = sorted([p for p in folder.glob("*") if p.is_file() and is_image_file(p)])
            if files:
                class_to_files[cls_str] = files

    return class_to_files

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def write_file_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def safe_sample(items: List[Path], k: int) -> List[Path]:
    if k < 0:
        raise ValueError("k no puede ser negativo")
    if k > len(items):
        raise ValueError(f"No hay suficientes elementos para muestrear: pedido={k}, disponibles={len(items)}")
    return random.sample(items, k)

def proportional_count(total: int, proportion: float) -> int:
    return int(round(total * proportion))

def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def copy_selected_images(
    selected_real: Dict[str, List[Path]],
    selected_synth: Dict[str, List[Path]],
    out_train_dir: Path
) -> None:
    ensure_dir(out_train_dir)
    todos_real = [(cls, i, src) for cls, lst in selected_real.items() for i, src in enumerate(lst, start=1)]
    todos_synth = [(cls, i, src) for cls, lst in selected_synth.items() for i, src in enumerate(lst, start=1)]

    for cls, i, src in tqdm(todos_real, desc=f"Copiando originales -> {out_train_dir.name}"):
        dst_name = f"real_{cls}_{i:06d}_{src.name}"
        write_file_copy(src, out_train_dir / dst_name)

    for cls, i, src in tqdm(todos_synth, desc=f"Copiando sintéticas -> {out_train_dir.name}"):
        dst_name = f"synth_{cls}_{i:06d}_{src.name}"
        write_file_copy(src, out_train_dir / dst_name)

def build_dataset_json(
    selected_real: Dict[str, List[Path]],
    selected_synth: Dict[str, List[Path]],
    out_dir: Path
):
    labels = []
    labels_bin = []

    for cls_str in sorted(set(selected_real.keys()) | set(selected_synth.keys())):
        cls_int = int(cls_str)
        bin_label = CLASS_TO_BIN.get(cls_int, -1)
        
        for i, src in enumerate(selected_real.get(cls_str, []), start=1):
            dst_name = f"real_{cls_str}_{i:06d}_{src.name}"
            labels.append([dst_name, cls_int])
            labels_bin.append([dst_name, bin_label])

        for i, src in enumerate(selected_synth.get(cls_str, []), start=1):
            dst_name = f"synth_{cls_str}_{i:06d}_{src.name}"
            labels.append([dst_name, cls_int])
            labels_bin.append([dst_name, bin_label])

    save_json(out_dir / "dataset.json", {"labels": labels})
    save_json(out_dir / "dataset_binary.json", {"labels": labels_bin})

def summarize_selection(
    selected_real: Dict[str, List[Path]],
    selected_synth: Dict[str, List[Path]],
    proportion_name: str,
    mode_name: str,
    target_per_class: Dict[str, int] = None
) -> dict:
    summary = {
        "mode": mode_name,
        "scenario": proportion_name,
        "classes": {},
        "global": {
            "real_total": 0,
            "synth_total": 0,
            "final_total": 0,
            "synthetic_fraction_global": 0.0
        }
    }

    all_classes = sorted(set(selected_real.keys()) | set(selected_synth.keys()), key=lambda x: int(x))
    for cls in all_classes:
        n_real = len(selected_real.get(cls, []))
        n_synth = len(selected_synth.get(cls, []))
        n_total = n_real + n_synth
        synth_frac = (n_synth / n_total) if n_total > 0 else 0.0

        summary["classes"][cls] = {
            "n_real": n_real,
            "n_synth": n_synth,
            "n_total": n_total,
            "synthetic_fraction": synth_frac
        }

        if target_per_class is not None:
            summary["classes"][cls]["target_per_class"] = target_per_class.get(cls)

        summary["global"]["real_total"] += n_real
        summary["global"]["synth_total"] += n_synth
        summary["global"]["final_total"] += n_total

    gt = summary["global"]["final_total"]
    summary["global"]["synthetic_fraction_global"] = (
        summary["global"]["synth_total"] / gt if gt > 0 else 0.0
    )
    return summary