from pathlib import Path
import hashlib
import shutil
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import imagehash


def export_images_without_duplicates(
    img_dir: str | Path,
    out_dir: str | Path,
    exts=(".jpg"),
    phash_threshold: int | None = 0,      # None = no pHash; 0 = rápido por grupos; >0 = O(N^2)
    use_sha256: bool = True,              # recomendado
    preserve_structure: bool = False,     # True: mantiene subcarpetas relativas
    dry_run: bool = False,                # True: no copia, solo reporta
    review_dir: str | Path | None = None  # carpeta para guardar (rep + removidas) por grupo
):
    """
    Exporta una copia del dataset sin duplicados.
    Incluye un reporte detallado en terminal y mantiene el resumen original.
    """

    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    if review_dir is not None:
        review_dir = Path(review_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"No existe img_dir: {img_dir}")

    # -------- helpers --------
    def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def phash_str(path: Path) -> str:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return str(imagehash.phash(im))

    def safe_copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            return dst
        stem, suf = dst.stem, dst.suffix
        k = 1
        while True:
            alt = dst.with_name(f"{stem}__{k}{suf}")
            if not alt.exists():
                shutil.copy2(src, alt)
                return alt
            k += 1

    def dst_path_for(src: Path, base_dir: Path) -> Path:
        if preserve_structure:
            rel = src.relative_to(img_dir)
            return base_dir / rel
        return base_dir / src.name

    # [NUEVO] Estructura para el reporte detallado: Rep -> Lista de duplicados
    all_dups_map = defaultdict(list)

    # -------- 1) listar imágenes --------
    paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    total_found = len(paths)
    if total_found == 0:
        raise ValueError(f"No se encontraron imágenes en {img_dir} con extensiones {sorted(exts)}")

    # -------- 2) dedup exacto (sha256) --------
    removed_sha = []
    keep_after_sha = paths

    if use_sha256:
        sha_to_rep = {}
        kept_paths_temp = [] # Lista temporal para mantener orden y filtrar

        for p in tqdm(paths, desc="SHA-256", leave=False):
            try:
                h = sha256_file(p)
                if h not in sha_to_rep:
                    sha_to_rep[h] = p
                    kept_paths_temp.append(p)
                else:
                    # Es un duplicado exacto
                    rep = sha_to_rep[h]
                    removed_sha.append(p)
                    # [NUEVO] Registrar en el mapa global
                    all_dups_map[rep].append(p)
            except Exception:
                pass
        keep_after_sha = kept_paths_temp

    # -------- 3) dedup visual (pHash) --------
    removed_phash = []
    keep_final = keep_after_sha

    # mapping para revisión: rep -> lista de removidas
    phash_groups_map = {}  # phash_value -> [paths...]
    rep_to_dups = defaultdict(list)

    if phash_threshold is not None:
        if phash_threshold == 0:
            # --- RÁPIDO: agrupar por pHash string ---
            groups = defaultdict(list)
            for p in tqdm(keep_after_sha, desc="pHash (group)", leave=False):
                try:
                    groups[phash_str(p)].append(p)
                except Exception:
                    pass

            phash_groups_map = dict(groups)

            keep_final = []
            for _, group_paths in phash_groups_map.items():
                # el primero es el representante
                rep = group_paths[0]
                keep_final.append(rep)
                # el resto se considera duplicado (pHash idéntico)
                for dup in group_paths[1:]:
                    removed_phash.append(dup)
                    rep_to_dups[rep].append(dup)
                    
                    # [NUEVO] Registrar y heredar duplicados previos
                    all_dups_map[rep].append(dup)
                    if dup in all_dups_map:
                        all_dups_map[rep].extend(all_dups_map[dup])
                        del all_dups_map[dup]

        else:
            # --- LENTO: O(N^2) para thresholds > 0 ---
            phashes = []
            kept_paths = []

            for p in tqdm(keep_after_sha, desc="Calc pHash", leave=False):
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        phashes.append(imagehash.phash(im))
                    kept_paths.append(p)
                except Exception:
                    pass

            visited = [False] * len(kept_paths)
            chosen = []

            for i in tqdm(range(len(kept_paths)), desc="Agrupando pHash", leave=False):
                if visited[i]:
                    continue
                rep = kept_paths[i]
                chosen.append(rep)
                visited[i] = True

                for j in range(i + 1, len(kept_paths)):
                    if visited[j]:
                        continue
                    dist = phashes[i] - phashes[j]
                    if dist <= phash_threshold:
                        visited[j] = True
                        dup = kept_paths[j]
                        removed_phash.append(dup)
                        rep_to_dups[rep].append(dup)

                        # [NUEVO] Registrar y heredar duplicados previos
                        all_dups_map[rep].append(dup)
                        if dup in all_dups_map:
                            all_dups_map[rep].extend(all_dups_map[dup])
                            del all_dups_map[dup]

            keep_final = chosen

    # -------- 4) copiar dataset final --------
    if not dry_run:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for p in tqdm(keep_final, desc="Copiando (final)"):
            try:
                safe_copy(p, dst_path_for(p, out_dir))
            except Exception:
                pass

    # -------- 5) carpeta de revisión (rep + removidas similares) --------
    if (review_dir is not None) and (not dry_run) and (phash_threshold is not None):
        review_dir.mkdir(parents=True, exist_ok=True)

        # una carpeta por grupo (solo grupos con removidas)
        gid = 0
        for rep, dups in rep_to_dups.items():
            gid += 1
            group_dir = review_dir / f"group_{gid:05d}"
            group_dir.mkdir(parents=True, exist_ok=True)

            # copia representante
            safe_copy(rep, group_dir / f"rep__{rep.name}")

            # copia duplicadas
            for k, dup in enumerate(dups, start=1):
                safe_copy(dup, group_dir / f"dup{k:02d}__{dup.name}")

    stats = {
        "total_found": total_found,
        "use_sha256": use_sha256,
        "phash_threshold": phash_threshold,
        "unique_after_sha": len(keep_after_sha),
        "removed_sha": len(removed_sha),
        "unique_after_final": len(keep_final),
        "removed_phash": len(removed_phash),
        "rep_to_dups": dict(rep_to_dups), 
    }

    print("\n" + "="*80)
    print("REPORTE DETALLADO (Archivo Conservado <--- Duplicados Eliminados)")
    print("="*80)
    
    dups_count_check = 0
    for rep in keep_final:
        dups = all_dups_map.get(rep, [])
        if dups:
            dups_names = ", ".join([d.name for d in dups])
            # Imprimimos alineado
            print(f"[KEEP] {rep.name:<30} <--- [DEL] {dups_names}")
            dups_count_check += len(dups)
    print("-" * 80 + "\n")

    print("=== RESUMEN ===")
    print(f"Total encontradas:            {stats['total_found']}")
    print(f"Total duplicadas/borradas: {dups_count_check}")
    if use_sha256:
        print(f"Únicas tras SHA-256:          {stats['unique_after_sha']}  (removidas: {stats['removed_sha']})")
    else:
        print("SHA-256:                      desactivado")
    if phash_threshold is None:
        print("pHash:                        desactivado")
    else:
        print(f"Únicas tras pHash(thr={phash_threshold}): {stats['unique_after_final']}  (removidas: {stats['removed_phash']})")
    if not dry_run:
        print(f"Salida sin duplicados:        {Path(out_dir).resolve()}")
        if review_dir is not None:
            print(f"Carpeta de revisión (grupos): {Path(review_dir).resolve()}")

    return stats