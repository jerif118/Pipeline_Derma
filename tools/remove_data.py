import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

EXT = ".jpg"           # si tus imágenes son jpg

# df_clean ya debería existir en tu notebook
# df_clean = df[df["diagnosis"] != "other"].copy()
def to_filename(x: str) -> str:
    x = str(x)
    return x if x.lower().endswith(EXT) else x + EXT

def remove(img_dir: str | Path, out_dir: str | Path,df:pd.DataFrame,col : str):
    
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"No existe img_dir: {img_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_files = [to_filename(x) for x in df[col].tolist()]

    copied, missing = 0, 0
    for fn in tqdm(keep_files, desc="Copiando imágenes filtradas"):
        src = img_dir / fn
        dst = out_dir / fn
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    print("Filas df_clean:", len(df))
    print("Imágenes copiadas:", copied)
    print("No encontradas:", missing)
    print("Carpeta salida:", out_dir.resolve())
