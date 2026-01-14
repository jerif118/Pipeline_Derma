import os
import json

# 1. Preparamos la lista de etiquetas en el formato de NVIDIA: ["nombre_archivo.jpg", label]
# Usamos 'label_esp' porque es la que define la patología exacta para la GAN
def labels_json(df,col_id,label,out_dir):
    labels = [
        [f"{row[col_id]}.jpg", int(row[label])] 
        for _, row in df.iterrows()
    ]

    # 2. Creamos el diccionario con la clave "labels" requerida por StyleGAN
    dataset_json_content = {
        "labels": labels
    }

    out_json = os.path.join(out_dir,"dataset.json")
    
    # 3. Guardamos el archivo
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(dataset_json_content, f, indent=4)
    print("Archivo creado.")