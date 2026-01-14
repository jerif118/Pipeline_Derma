import os
import pandas as pd

def sincronizar_(df: pd.DataFrame, img_dir: str, col:str, extension='.jpg'):
    """
    Filtra el DataFrame para conservar solo las filas cuyas imágenes 
    existen físicamente en el disco.
    """
    # 1. Obtener el set de archivos reales en la carpeta (operación O(1))
    archivos_en_disco = set(os.listdir(img_dir))
    
    # 2. Asegurarnos de que el ID en el DF tenga la extensión para comparar
    # Si tus IDs ya tienen .jpg, esto no hará nada. Si no, se lo agrega temporalmente.
    def tiene_extension(nombre):
        return str(nombre).lower().endswith(extension)
    
    # Creamos una máscara booleana
    mascara = df[col].apply(
        lambda x: x if tiene_extension(x) else f"{x}{extension}"
    ).isin(archivos_en_disco)
    
    df_resultante = df[mascara].copy()
    
    eliminados = len(df) - len(df_resultante)
    print(f"--- Reporte de Sincronización ---")
    print(f"Filas originales: {len(df)}")
    print(f"Filas conservadas: {len(df_resultante)}")
    print(f"Filas eliminadas (no encontradas en disco): {eliminados}")
    print(f"---------------------------------")
    
    return df_resultante