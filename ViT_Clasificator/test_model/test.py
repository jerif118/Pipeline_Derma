import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data.transform import transforms_gpu_val

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model, dataloader):
    model.to(device)
    model.eval()
    criterion_bin = torch.nn.BCEWithLogitsLoss(reduction='none')
    criterion_class = torch.nn.CrossEntropyLoss()
    transform_test = transforms_gpu_val()

    all_y_bin, all_y_bin_hat, all_y_bin_proba = [], [], []
    all_y_class, all_y_class_hat, all_y_class_proba = [], [], []
    test_loss_bin, test_loss_class = [], []

    bar = tqdm(dataloader, desc="Testing")
    with torch.no_grad():
        for batch in bar:
            X, y_bin, y_class = batch
            X, y_bin, y_class = X.to(device), y_bin.to(device), y_class.to(device)
            X = transform_test(X)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                y_bin_hat, y_class_hat = model(X)
                valid_mask = (y_bin != -1)
                loss_bin = criterion_bin(y_bin_hat.squeeze(1), y_bin)
                loss_bin = loss_bin[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0).to(device)
                loss_class = criterion_class(y_class_hat, y_class)
                test_loss_bin.append(loss_bin.item())
                test_loss_class.append(loss_class.item())
                
            bin_logits = y_bin_hat.squeeze(1).detach().float()
            class_logits = y_class_hat.detach().float()
            bin_proba = torch.sigmoid(bin_logits).cpu()
            class_proba = torch.softmax(class_logits, dim=1).cpu()

            if valid_mask.sum() > 0:
                all_y_bin.append(y_bin[valid_mask].cpu())
                all_y_bin_hat.append((y_bin_hat.squeeze(1)[valid_mask] > 0).long().cpu())
                all_y_bin_proba.append(bin_proba[valid_mask.cpu()])
            all_y_class.append(y_class.cpu())
            all_y_class_hat.append(torch.argmax(y_class_hat, dim=1).cpu())
            all_y_class_proba.append(class_proba)

    all_y_bin = torch.cat(all_y_bin).cpu().numpy() if all_y_bin else np.array([])
    all_y_bin_hat = torch.cat(all_y_bin_hat).cpu().numpy() if all_y_bin_hat else np.array([])
    all_y_bin_proba = torch.cat(all_y_bin_proba).cpu().numpy() if all_y_bin_proba else np.array([])
    all_y_class = torch.cat(all_y_class).cpu().numpy()
    all_y_class_hat = torch.cat(all_y_class_hat).cpu().numpy()
    all_y_class_proba = torch.cat(all_y_class_proba).cpu().numpy()

    acc_bin = accuracy_score(all_y_bin, all_y_bin_hat) if len(all_y_bin) > 0 else 0.0
    acc_class = accuracy_score(all_y_class, all_y_class_hat)

    results = {
        "test_loss_bin": float(np.mean(test_loss_bin)),
        "test_loss_class": float(np.mean(test_loss_class)),
        "test_acc_bin": float(acc_bin),
        "test_acc_class": float(acc_class),
    }

    print(f"Test binary: loss {results['test_loss_bin']:.5f} acc {results['test_acc_bin']:.5f}, "
          f"class: loss {results['test_loss_class']:.5f} acc {results['test_acc_class']:.5f}")

    return {
        "results": results,
        "y_bin": all_y_bin,
        "y_bin_hat": all_y_bin_hat,
        "y_bin_proba": all_y_bin_proba,
        "y_class": all_y_class,
        "y_class_hat": all_y_class_hat,
        "y_class_proba": all_y_class_proba,
    }
