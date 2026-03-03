import os, json

def load_labels(dir,file):
    js_path = os.path.join(dir, file)
    with open(js_path,"r") as f:
        d = json.load(f)
    return d["labels"]