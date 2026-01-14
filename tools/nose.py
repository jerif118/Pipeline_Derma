import os
import shutil
from tqdm import tqdm

def organizar_dataset(df,img_dir:str, dst_path:str,col_id:str ):
    """
    Copia las imágenes a una estructura de carpetas: base_path/subset_name/clase/imagen.jpg
    """
    print(f"Organizando set de {dst_path}...")
    
    #img_dir = Path(img_dir)
    #out_dir = Path(out_dir)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Definir origen (ajusta 'carpeta_original' a tu ruta real)
        img_name = f"{row[col_id]}.jpg"
        src_path = os.path.join(img_dir, img_name)
        os.makedirs(dst_path, exist_ok=True)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    print("imagenes copiadas correctamente")

