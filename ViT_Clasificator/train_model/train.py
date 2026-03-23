import torch
import numpy as np
import csv
import os
from tqdm import tqdm
from data.transform import transforms_gpu_train, transforms_gpu_val

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fit(model, dataloader, epochs=5, lr=0.001, w_bin=0.5, w_class=0.5, max_norm=1.0, log_path="training_log.csv", class_weights=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_bin = torch.nn.BCEWithLogitsLoss(reduction='none')
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion_class = torch.nn.CrossEntropyLoss(weight=class_weights)
    transform_train = transforms_gpu_train()
    transform_val = transforms_gpu_val()
    history = []
    for epoch in range(1,epochs+1):
        model.train()
        train_loss_bin,train_loss_class, train_acc_bin, train_acc_class = [], [], [],[]
        bar = tqdm(dataloader['train'])
        for batch in bar:
            X, y_bin, y_class = batch
            X, y_bin, y_class = X.to(device,non_blocking=True), y_bin.to(device, non_blocking=True), y_class.to(device, non_blocking=True)
            X =transform_train(X)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
                y_bin_hat, y_class_hat = model(X)
                valid_mask = (y_bin != -1)
                loss_bin = criterion_bin(y_bin_hat.squeeze(1), y_bin)
                loss_bin = loss_bin[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0).to(device)
                loss_class = criterion_class(y_class_hat, y_class)
                loss = (w_bin*loss_bin)+(w_class*loss_class)
                if valid_mask.sum()>0:
                    acc_bin = ((y_bin[valid_mask] == (y_bin_hat.squeeze(1)[valid_mask] > 0)).sum().item() / valid_mask.sum().item())
                else:
                    acc_bin = 0.0
                acc_class = (y_class == torch.argmax(y_class_hat, axis=1)).sum().item()/len(y_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            train_loss_bin.append(loss_bin.item())
            train_loss_class.append(loss_class.item())
            train_acc_bin.append(acc_bin)
            train_acc_class.append(acc_class)
            bar.set_description(f"binary: loss {np.mean(train_loss_bin):.5f} acc{np.mean(train_acc_bin):.5f}, class: loss {np.mean(train_loss_class):.5f} acc{np.mean(train_acc_class):.5f}")
        bar = tqdm(dataloader['val'])
        val_loss_bin, val_loss_class, val_acc_bin, val_acc_class = [], [], [], []
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y_bin, y_class = batch
                X, y_bin, y_class = X.to(device, non_blocking=True), y_bin.to(device, non_blocking=True), y_class.to(device, non_blocking=True)
                X = transform_val(X)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
                    y_bin_hat, y_class_hat = model(X)
                    valid_mask = (y_bin != -1)
                    loss_bin = criterion_bin(y_bin_hat.squeeze(1), y_bin)
                    loss_bin = loss_bin[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0).to(device)
                    loss_class = criterion_class(y_class_hat, y_class)
                    val_loss_bin.append(loss_bin.item())
                    val_loss_class.append(loss_class.item())
                    if valid_mask.sum()>0:
                        acc_bin = ((y_bin[valid_mask] == (y_bin_hat.squeeze(1)[valid_mask] > 0)).sum().item() / valid_mask.sum().item())
                    else:
                        acc_bin = 0.0
                    acc_class = (y_class == torch.argmax(y_class_hat, axis=1)).sum().item()/len(y_class)
                val_acc_bin.append(acc_bin)
                val_acc_class.append(acc_class)
                bar.set_description(f"binary: loss {np.mean(val_loss_bin):.5f} acc{np.mean(val_acc_bin):.5f}, class: loss {np.mean(val_loss_class):.5f} acc{np.mean(val_acc_class):.5f}")
        row = {
            "epoch": epoch,
            "train_loss_bin": np.mean(train_loss_bin),
            "train_loss_class": np.mean(train_loss_class),
            "train_acc_bin": np.mean(train_acc_bin),
            "train_acc_class": np.mean(train_acc_class),
            "val_loss_bin": np.mean(val_loss_bin),
            "val_loss_class": np.mean(val_loss_class),
            "val_acc_bin": np.mean(val_acc_bin),
            "val_acc_class": np.mean(val_acc_class),
        }
        history.append(row)
        print(f"Epoch {epoch}/{epochs} binary: loss {row['train_loss_bin']:.5f} val_loss {row['val_loss_bin']:.5f} acc {row['train_acc_bin']:.5f} val_acc {row['val_acc_bin']:.5f}, class: loss {row['train_loss_class']:.5f} val_loss {row['val_loss_class']:.5f} acc {row['train_acc_class']:.5f} val_acc {row['val_acc_class']:.5f}")
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(history)
    return history