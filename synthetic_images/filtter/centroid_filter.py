import shutil
from pathlib import Path

import numpy as np
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .inception_embedder import build_inception_embedder
from .inception_embedder import extract_embeddings
from .inception_embedder import load_real_paths_by_class


def compute_iqr_threshold(values, upper=True, iqr_multiplier=1.5):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "threshold": float("nan"),
            "mode": "iqr",
        }
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    threshold = q3 + iqr_multiplier * iqr if upper else q1 - iqr_multiplier * iqr
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "threshold": float(threshold),
        "mode": "iqr",
    }


def compute_quantile_threshold(values, quantile=0.95):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "threshold": float("nan"),
            "mode": "quantile",
            "quantile": float(quantile),
        }

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    threshold = np.percentile(values, quantile * 100)

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "threshold": float(threshold),
        "mode": "quantile",
        "quantile": float(quantile),
    }


def compute_real_threshold_stats(
    real_distances,
    threshold_mode="iqr",
    iqr_multiplier=1.5,
    quantile=0.95,
):
    iqr_stats = compute_iqr_threshold(real_distances, upper=True, iqr_multiplier=iqr_multiplier)

    if threshold_mode == "iqr":
        return iqr_stats

    quantile_stats = compute_quantile_threshold(real_distances, quantile=quantile)
    if threshold_mode == "quantile":
        return quantile_stats

    if threshold_mode == "hybrid":
        # Regla más estricta para evitar cortes demasiado permisivos por clase.
        threshold = min(iqr_stats["threshold"], quantile_stats["threshold"])
        return {
            "q1": iqr_stats["q1"],
            "q3": iqr_stats["q3"],
            "iqr": iqr_stats["iqr"],
            "threshold": float(threshold),
            "mode": "hybrid",
            "iqr_threshold": iqr_stats["threshold"],
            "quantile_threshold": quantile_stats["threshold"],
            "quantile": float(quantile),
        }

    raise ValueError("threshold_mode must be one of: 'iqr', 'quantile', 'hybrid'")


def split_items_by_iqr(
    items,
    value_key,
    upper=True,
    threshold=None,
    adaptive_quantile=None,
    combine_mode="min",
):
    if threshold is None and items:
        threshold = items[0].get("real_threshold")

    values = np.array([item[value_key] for item in items], dtype=float) if items else np.array([], dtype=float)

    if threshold is None:
        stats = compute_iqr_threshold(values, upper=upper)
    else:
        first_item = items[0] if items else {}
        stats = {
            "q1": float(first_item.get("real_q1", float("nan"))),
            "q3": float(first_item.get("real_q3", float("nan"))),
            "iqr": float(first_item.get("real_iqr", float("nan"))),
            "threshold": float(threshold),
            "mode": "real_threshold",
        }

    # Mezcla opcional con un percentil de la distribución sintética de la clase.
    # Para upper=True, usar min() vuelve el filtro más estricto si el umbral real es laxo.
    if adaptive_quantile is not None and values.size > 0:
        if not 0.0 < adaptive_quantile < 1.0:
            raise ValueError("adaptive_quantile must be in (0, 1)")

        synth_threshold = float(np.percentile(values, adaptive_quantile * 100.0))
        base_threshold = float(stats["threshold"])

        if combine_mode == "min":
            combined_threshold = min(base_threshold, synth_threshold) if upper else max(base_threshold, synth_threshold)
        elif combine_mode == "max":
            combined_threshold = max(base_threshold, synth_threshold) if upper else min(base_threshold, synth_threshold)
        else:
            raise ValueError("combine_mode must be either 'min' or 'max'")

        stats["base_threshold"] = base_threshold
        stats["synthetic_quantile"] = float(adaptive_quantile)
        stats["synthetic_threshold"] = synth_threshold
        stats["threshold"] = float(combined_threshold)
        stats["mode"] = f"adaptive_{combine_mode}"

    if upper:
        kept = [item for item in items if item[value_key] <= stats["threshold"]]
        discarded = [item for item in items if item[value_key] > stats["threshold"]]
    else:
        kept = [item for item in items if item[value_key] >= stats["threshold"]]
        discarded = [item for item in items if item[value_key] < stats["threshold"]]

    return kept, discarded, stats


def compute_real_centroid_distances(
    train_dir,
    synthetic_base_dir,
    train_labels_path=None,
    class_indices=None,
    batch_size=32,
    pattern="*.png",
    threshold_mode="iqr",
    iqr_multiplier=1.5,
    quantile=0.95,
):
    model, preprocess, device = build_inception_embedder()
    real_paths_by_class = load_real_paths_by_class(train_dir, train_labels_path)
    synthetic_base_dir = Path(synthetic_base_dir)

    if class_indices is None:
        class_indices = sorted(real_paths_by_class.keys())

    all_distances = {}
    real_centroids = {}

    for class_index in class_indices:
        real_paths = real_paths_by_class.get(int(class_index), [])
        real_embeddings, valid_real_paths = extract_embeddings(
            real_paths,
            model,
            preprocess,
            device,
            batch_size=batch_size,
        )
        if len(real_embeddings) == 0:
            continue

        try:
            pca = PCA(n_components=0.95)
            real_embeddings_reduced = pca.fit_transform(real_embeddings)
        except ValueError:
            pca = None
            real_embeddings_reduced = real_embeddings

        max_k = min(10, len(real_embeddings_reduced))
        k_values = list(range(1, max_k + 1))
        inertias = []

        for k in k_values:
            model_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            model_kmeans.fit(real_embeddings_reduced)
            inertias.append(float(model_kmeans.inertia_))

        if len(k_values) == 1:
            optimal_k = 1
        else:
            knee = KneeLocator(
                k_values,
                inertias,
                curve="convex",
                direction="decreasing",
            )
            optimal_k = int(knee.elbow or knee.knee or 1)

        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_kmeans.fit(real_embeddings_reduced)
        sub_centroids = final_kmeans.cluster_centers_

        real_distance_matrix = cdist(
            real_embeddings_reduced,
            sub_centroids,
            metric="cosine",
        )
        real_min_distances = real_distance_matrix.min(axis=1)
        real_stats = compute_real_threshold_stats(
            real_min_distances,
            threshold_mode=threshold_mode,
            iqr_multiplier=iqr_multiplier,
            quantile=quantile,
        )

        real_centroids[int(class_index)] = sub_centroids
        real_paths_by_class[int(class_index)] = valid_real_paths

        synthetic_dir = synthetic_base_dir / f"imagenes_{class_index}"
        synthetic_paths = sorted(synthetic_dir.glob(pattern))
        synthetic_embeddings, valid_synthetic_paths = extract_embeddings(
            synthetic_paths,
            model,
            preprocess,
            device,
            batch_size=batch_size,
        )

        if len(synthetic_embeddings) == 0:
            all_distances[int(class_index)] = []
            continue

        if pca is not None:
            synthetic_embeddings_reduced = pca.transform(synthetic_embeddings)
        else:
            synthetic_embeddings_reduced = synthetic_embeddings

        synthetic_distance_matrix = cdist(
            synthetic_embeddings_reduced,
            sub_centroids,
            metric="cosine",
        )
        distances = synthetic_distance_matrix.min(axis=1)

        all_distances[int(class_index)] = [
            {
                "path": str(path),
                "filename": path.name,
                "distance": float(distance),
                "real_q1": real_stats["q1"],
                "real_q3": real_stats["q3"],
                "real_iqr": real_stats["iqr"],
                "real_threshold": real_stats["threshold"],
                "n_subcentroids": int(optimal_k),
                "threshold_mode": real_stats.get("mode", "iqr"),
            }
            for path, distance in zip(valid_synthetic_paths, distances)
        ]

    return real_paths_by_class, real_centroids, all_distances

def build_distance_summary(
    all_distances,
    real_paths_by_class,
    adaptive_quantile=None,
    combine_mode="min",
):
    rows = []
    for class_index in sorted(all_distances.keys()):
        items = all_distances[class_index]
        if not items:
            continue
        distances = np.array([item["distance"] for item in items], dtype=float)
        kept, discarded, stats = split_items_by_iqr(
            items,
            "distance",
            upper=True,
            threshold=items[0].get("real_threshold"),
            adaptive_quantile=adaptive_quantile,
            combine_mode=combine_mode,
        )
        rows.append(
            {
                "Clase": class_index,
                "Reales": len(real_paths_by_class.get(class_index, [])),
                "Sintéticas": len(items),
                "Retenidas": len(kept),
                "Descartadas": len(discarded),
                "% Descartado": round(len(discarded) / len(items) * 100, 1),
                "Mediana distancia": round(float(np.median(distances)), 4),
                "IQR": round(stats["iqr"], 4),
                "Corte": round(stats["threshold"], 4),
                "Corte base": round(float(stats.get("base_threshold", stats["threshold"])), 4),
                "Corte sintético": round(float(stats.get("synthetic_threshold", float("nan"))), 4),
                "Modo corte": stats.get("mode", "n/a"),
            }
        )
    return rows


def copy_filtered_images(
    all_distances,
    synthetic_base_dir,
    suffix_good="_ok",
    suffix_bad="_out",
    adaptive_quantile=None,
    combine_mode="min",
):
    synthetic_base_dir = Path(synthetic_base_dir)
    results = []

    for class_index, items in all_distances.items():
        synthetic_dir = synthetic_base_dir / f"imagenes_{class_index}"
        kept, discarded, stats = split_items_by_iqr(
            items,
            "distance",
            upper=True,
            threshold=items[0].get("real_threshold") if items else None,
            adaptive_quantile=adaptive_quantile,
            combine_mode=combine_mode,
        )

        out_good = synthetic_dir.parent / f"{synthetic_dir.name}{suffix_good}"
        out_bad = synthetic_dir.parent / f"{synthetic_dir.name}{suffix_bad}"
        out_good.mkdir(parents=True, exist_ok=True)
        out_bad.mkdir(parents=True, exist_ok=True)

        for item in kept:
            src = Path(item["path"])
            shutil.copy2(src, out_good / src.name)

        for item in discarded:
            src = Path(item["path"])
            shutil.copy2(src, out_bad / src.name)

        results.append(
            {
                "class_index": class_index,
                "kept": len(kept),
                "discarded": len(discarded),
                "threshold": stats["threshold"],
                "base_threshold": stats.get("base_threshold", stats["threshold"]),
                "synthetic_threshold": stats.get("synthetic_threshold"),
                "threshold_mode": stats.get("mode", "n/a"),
                "out_good": out_good,
                "out_bad": out_bad,
            }
        )

    return results
