import os
import json
def labels_json(df,col_id,label,out_dir,name="dataset.json"):
    labels = [
        [f"{row[col_id]}.jpg", int(row[label])] 
        for _, row in df.iterrows()
    ]
    dataset_json_content = {
        "labels": labels
    }

    out_json = os.path.join(out_dir,name)
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(dataset_json_content, f, indent=4)
    print("Archivo creado.")