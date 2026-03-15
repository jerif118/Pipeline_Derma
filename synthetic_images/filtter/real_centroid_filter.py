"""Compatibilidad para imports antiguos.

El flujo actual se dividió entre `inception_embedder.py` y `centroid_filter.py`.
Este archivo se mantiene solo para no romper imports existentes.
"""

from .centroid_filter import build_distance_summary
from .centroid_filter import compute_iqr_threshold
from .centroid_filter import compute_real_centroid_distances
from .centroid_filter import copy_filtered_images
from .centroid_filter import split_items_by_iqr
from .inception_embedder import build_inception_embedder
from .inception_embedder import extract_embeddings
from .inception_embedder import load_real_paths_by_class