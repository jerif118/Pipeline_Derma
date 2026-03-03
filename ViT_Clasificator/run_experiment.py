import argparse
import os
import json
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.model import Model
from data.data import Dataset
from data.transform import transforms_train, transforms_val
from train_model.train import fit
from test_model.test import test
from test_model.evaluate import evaluate, save_results
from utils.utils import load_labels
from stats.statistics import (
    cochrans_q,
    mcnemar_posthoc_corrected,
    scalar_bootstrap_ci,
    compare_scenarios,
)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single(run_id, args, train_dir, output_dir):
    seed = args.base_seed + run_id
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"RUN {run_id + 1}/{args.n_runs}  (seed={seed})")
    print(f"{'='*60}")

    # --- Datos train/val ---
    train_labels = load_labels(train_dir, args.file_json)
    train_labels_bin = load_labels(train_dir, args.file_binary_json)
    val_labels = load_labels(args.val_dir, args.file_json)
    val_labels_bin = load_labels(args.val_dir, args.file_binary_json)

    train_t = transforms_train()
    val_t = transforms_val()

    train_ds = Dataset(train_dir, train_labels, train_labels_bin, transform=train_t)
    val_ds = Dataset(args.val_dir, val_labels, val_labels_bin, transform=val_t)

    dataloader = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True),
    }

    # --- Class weights (inverse frequency) para CrossEntropyLoss ---
    class_counts = np.bincount([label for _, label in train_labels], minlength=args.n_outputs)
    class_counts = np.maximum(class_counts, 1)  # evitar división por cero
    class_weights = 1.0 / class_counts.astype(float)
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalizar

    # --- Modelo ---
    model = Model(n_binary=1, n_outputs=args.n_outputs, freeze=args.freeze)

    # --- Entrenamiento ---
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "training_log.csv")

    history = fit(
        model, dataloader,
        epochs=args.epochs,
        lr=args.lr,
        w_bin=args.w_bin,
        w_class=args.w_class,
        max_norm=args.max_norm,
        log_path=log_path,
        class_weights=class_weights.tolist(),
    )

    plot_training_curves(history, run_dir)

    # --- Guardar history de entrenamiento ---
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # --- Guardar pesos ---
    weights_path = os.path.join(run_dir, "model.pth")
    torch.save(model.state_dict(), weights_path)

    # --- Datos test ---
    test_labels = load_labels(args.test_dir, args.file_json)
    test_labels_bin = load_labels(args.test_dir, args.file_binary_json)
    test_ds = Dataset(args.test_dir, test_labels, test_labels_bin, transform=val_t)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Test ---
    test_out = test(model, test_loader)

    # --- Métricas detalladas ---
    eval_metrics = evaluate(
        test_out["y_bin"], test_out["y_bin_hat"], test_out["y_bin_proba"],
        test_out["y_class"], test_out["y_class_hat"], test_out["y_class_proba"],
    )
    save_results(test_out["results"], eval_metrics, run_dir, run_id=run_id)

    # Guardar predicciones raw para análisis estadístico posterior
    np.savez(
        os.path.join(run_dir, "predictions.npz"),
        y_bin=test_out["y_bin"],
        y_bin_hat=test_out["y_bin_hat"],
        y_bin_proba=test_out["y_bin_proba"],
        y_class=test_out["y_class"],
        y_class_hat=test_out["y_class_hat"],
        y_class_proba=test_out["y_class_proba"],
    )

    return {
        "results": test_out["results"],
        "eval_metrics": eval_metrics,
        "history": history,
        "y_class": test_out["y_class"],
        "y_class_hat": test_out["y_class_hat"],
        "y_class_proba": test_out["y_class_proba"],
        "y_bin": test_out["y_bin"],
        "y_bin_hat": test_out["y_bin_hat"],
        "y_bin_proba": test_out["y_bin_proba"],
    }


def aggregate_results(all_run_data, output_dir):
    """Calcula media, std y Bootstrap CI de las N corridas."""
    all_results = [r["results"] for r in all_run_data]
    all_eval = [r["eval_metrics"] for r in all_run_data]

    # ── Métricas escalares ──
    # Combino results + metrics_class + metrics_bin
    scalar_keys = list(all_results[0].keys())
    class_keys = list(all_eval[0]["metrics_class"].keys())
    bin_keys = list(all_eval[0]["metrics_bin"].keys()) if all_eval[0]["metrics_bin"] else []

    stats_summary = {}

    for k in scalar_keys:
        values = [r[k] for r in all_results]
        stats_summary[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    for k in class_keys:
        values = [e["metrics_class"][k] for e in all_eval]
        stats_summary[f"class_{k}"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    for k in bin_keys:
        values = [e["metrics_bin"][k] for e in all_eval if e["metrics_bin"]]
        if values:
            stats_summary[f"bin_{k}"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    # ── Bootstrap CI (sobre los N valores escalares de cada métrica) ──
    # Cada corrida produce un valor escalar -> bootstrap sobre esos N valores.
    # Esto es metodológicamente correcto: no viola independencia entre muestras.
    boot_ci_class = {}
    for k in class_keys:
        values = [e["metrics_class"][k] for e in all_eval]
        boot_ci_class[k] = scalar_bootstrap_ci(values)

    boot_ci_bin = {}
    if bin_keys:
        for k in bin_keys:
            values = [e["metrics_bin"][k] for e in all_eval if e["metrics_bin"]]
            if values:
                boot_ci_bin[k] = scalar_bootstrap_ci(values)

    # JSON
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
            "metrics": stats_summary,
            "bootstrap_ci_class": boot_ci_class,
            "bootstrap_ci_bin": boot_ci_bin,
        }, f, indent=2)

    # CSV
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for k, v in stats_summary.items():
            writer.writerow([k, f"{v['mean']:.5f}", f"{v['std']:.5f}", f"{v['min']:.5f}", f"{v['max']:.5f}"])

    print(f"\n{'='*60}")
    print(f"RESUMEN ({len(all_results)} corridas)")
    print(f"{'='*60}")
    for k, v in stats_summary.items():
        print(f"  {k}: {v['mean']:.5f} ± {v['std']:.5f}  (min={v['min']:.5f}, max={v['max']:.5f})")
    print(f"\nBootstrap 95% CI — Multiclass (sobre {len(all_results)} valores escalares):")
    for m, ci in boot_ci_class.items():
        print(f"  {m}: [{ci['ci_lower']:.5f}, {ci['ci_upper']:.5f}]")
    if boot_ci_bin:
        print(f"\nBootstrap 95% CI — Binario (sobre {len(all_results)} valores escalares):")
        for m, ci in boot_ci_bin.items():
            print(f"  {m}: [{ci['ci_lower']:.5f}, {ci['ci_upper']:.5f}]")

    return stats_summary, boot_ci_class, boot_ci_bin

# run_experiment.py (agrega función)
def plot_training_curves(history, run_dir):
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(epochs, [h["train_loss_class"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss_class"] for h in history], label="val")
    axes[0].set_title("Class Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc_class"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc_class"] for h in history], label="val")
    axes[1].set_title("Class Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, [h["train_loss_bin"] for h in history], label="train")
    axes[2].plot(epochs, [h["val_loss_bin"] for h in history], label="val")
    axes[2].set_title("Binary Loss")
    axes[2].legend()

    axes[3].plot(epochs, [h["train_acc_bin"] for h in history], label="train")
    axes[3].plot(epochs, [h["val_acc_bin"] for h in history], label="val")
    axes[3].set_title("Binary Accuracy")
    axes[3].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
    plt.close(fig)



def run_statistical_tests(scenario_data: dict[str, list[dict]], output_dir: str):
    """
    Ejecuta pruebas estadísticas entre escenarios:
      - Cochran's Q + McNemar post-hoc con Holm-Bonferroni (binario por muestra)
      - Paired t-test / Wilcoxon (Balanced Accuracy entre escenarios)
    scenario_data: {scenario_name: [run_data_0, ..., run_data_N]}
    """
    scenario_names = list(scenario_data.keys())
    if len(scenario_names) < 2:
        print("Solo 1 escenario — no se ejecutan comparaciones estadísticas.")
        return

    stats_dir = os.path.join(output_dir, "statistical_tests")
    os.makedirs(stats_dir, exist_ok=True)

    # ── 1. Balanced Accuracy: paired comparison entre escenarios ──
    ba_per_scenario = {}
    for name, runs in scenario_data.items():
        ba_per_scenario[name] = np.array([
            r["eval_metrics"]["metrics_class"]["balanced_accuracy"] for r in runs
        ])

    pair_results = compare_scenarios(ba_per_scenario)
    with open(os.path.join(stats_dir, "paired_comparisons_ba.json"), "w") as f:
        json.dump(pair_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("COMPARACIONES PAREADAS (Balanced Accuracy)")
    print(f"{'='*60}")
    for pair, res in pair_results.items():
        sig = "**SIG**" if res["significant"] else "n.s."
        print(f"  {pair}: {res['test']} p={res['p_value']:.5f} {sig}")

    # ── 2. Cochran's Q + McNemar POR CORRIDA (no concatenado) ──
    # Cada corrida es un test independiente sobre 2638 muestras.
    # Reportamos: significativo en X/N corridas.
    n_runs = len(scenario_data[scenario_names[0]])
    if any(len(runs) != n_runs for runs in scenario_data.values()):
        raise ValueError("Todos los escenarios deben tener el mismo número de corridas.")

    q_results_per_run = []
    mcnemar_results_per_run = []

    for run_idx in range(n_runs):
        y_true_ref = scenario_data[scenario_names[0]][run_idx]["y_class"]
        correct_cols = []
        for name in scenario_names:
            run_data = scenario_data[name][run_idx]
            y_true = run_data["y_class"]
            y_hat = run_data["y_class_hat"]
            if y_true.shape != y_true_ref.shape or not np.array_equal(y_true, y_true_ref):
                raise ValueError(
                    f"Etiquetas y_class desalineadas entre escenarios en run={run_idx}, escenario={name}."
                )
            correct_cols.append((y_hat == y_true_ref).astype(int))
        run_correct_matrix = np.column_stack(correct_cols)  # (n_samples_test, k_scenarios)

        # Cochran's Q para esta corrida
        q_run = cochrans_q(run_correct_matrix)
        q_run["run"] = run_idx
        q_run["n_samples"] = int(run_correct_matrix.shape[0])
        q_results_per_run.append(q_run)

        # McNemar post-hoc para esta corrida (solo si Q es significativo)
        if q_run["significant"]:
            posthoc_run = mcnemar_posthoc_corrected(run_correct_matrix, labels=scenario_names)
            posthoc_run["run"] = run_idx
            mcnemar_results_per_run.append(posthoc_run)

    # Resumen Cochran's Q
    n_q_sig = sum(1 for q in q_results_per_run if q["significant"])
    q_p_values = [q["p_value"] for q in q_results_per_run]
    q_summary = {
        "n_runs": n_runs,
        "n_significant": n_q_sig,
        "fraction_significant": f"{n_q_sig}/{n_runs}",
        "p_values": q_p_values,
        "median_p": float(np.median(q_p_values)),
        "per_run": q_results_per_run,
    }
    with open(os.path.join(stats_dir, "cochrans_q.json"), "w") as f:
        json.dump(q_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("COCHRAN'S Q TEST (por corrida)")
    print(f"{'='*60}")
    for q in q_results_per_run:
        sig = "SÍ" if q["significant"] else "NO"
        print(f"  Run {q['run']}: Q={q['statistic']:.3f}, p={q['p_value']:.5f}, sig={sig}")
    print(f"  → Significativo en {n_q_sig}/{n_runs} corridas (mediana p={np.median(q_p_values):.5f})")

    # Resumen McNemar post-hoc
    if mcnemar_results_per_run:
        import pandas as pd
        all_posthoc = pd.concat(mcnemar_results_per_run, ignore_index=True)
        all_posthoc.to_csv(os.path.join(stats_dir, "mcnemar_posthoc_per_run.csv"), index=False)

        # Resumir por par: en cuántas corridas fue significativo
        from itertools import combinations
        print(f"\nMcNEMAR POST-HOC (Holm-Bonferroni, por corrida)")
        print(f"{'='*60}")
        pair_summary = {}
        for i, j in combinations(range(len(scenario_names)), 2):
            a, b = scenario_names[i], scenario_names[j]
            pair_rows = all_posthoc[
                (all_posthoc["model_a"] == a) & (all_posthoc["model_b"] == b)
            ]
            n_sig = int(pair_rows["significant"].sum())
            median_p = float(pair_rows["p_adjusted"].median()) if len(pair_rows) > 0 else float("nan")
            pair_summary[f"{a}_vs_{b}"] = {
                "n_significant": n_sig,
                "total_runs_tested": len(pair_rows),
                "fraction": f"{n_sig}/{len(pair_rows)}",
                "median_p_adjusted": median_p,
            }
            print(f"  {a} vs {b}: significativo en {n_sig}/{len(pair_rows)} corridas "
                  f"(mediana p_adj={median_p:.5f})")

        with open(os.path.join(stats_dir, "mcnemar_posthoc_summary.json"), "w") as f:
            json.dump(pair_summary, f, indent=2)
    else:
        print("\n  Cochran's Q no fue significativo en ninguna corrida → no se ejecutó McNemar.")

    # ── 3. Bootstrap CI por escenario (sobre N valores escalares) ──
    class_metric_keys = list(scenario_data[scenario_names[0]][0]["eval_metrics"]["metrics_class"].keys())
    boot_results = {}
    for name in scenario_names:
        runs = scenario_data[name]
        boot_results[name] = {}
        for mk in class_metric_keys:
            values = [r["eval_metrics"]["metrics_class"][mk] for r in runs]
            boot_results[name][mk] = scalar_bootstrap_ci(values)

    with open(os.path.join(stats_dir, "bootstrap_ci_per_scenario.json"), "w") as f:
        json.dump(boot_results, f, indent=2)

    print(f"\nBootstrap 95% CI por escenario (sobre {n_runs} valores escalares):")
    for name, metrics_ci in boot_results.items():
        ba = metrics_ci["balanced_accuracy"]
        print(f"  {name}: BA=[{ba['ci_lower']:.5f}, {ba['ci_upper']:.5f}]")


def parse_scenarios(scenario_args: list[str] | None) -> dict[str, str]:
    """Parsea argumentos de escenario con formato 'nombre=ruta'."""
    if not scenario_args:
        return {}
    scenarios = {}
    for s in scenario_args:
        if "=" not in s:
            raise ValueError(f"Formato de escenario inválido: '{s}'. Usa 'nombre=ruta'.")
        name, path = s.split("=", 1)
        scenarios[name] = path
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Experimento de N repeticiones train+test con análisis estadístico")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--w_bin", type=float, default=0.5)
    parser.add_argument("--w_class", type=float, default=0.5)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--n_outputs", type=int, default=11)
    parser.add_argument("--freeze", action="store_true", default=True)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--train_dir", type=str, default="../train",
                        help="Directorio de train por defecto (escenario control)")
    parser.add_argument("--val_dir", type=str, default="../validacion")
    parser.add_argument("--test_dir", type=str, default="../test")
    parser.add_argument("--file_json", type=str, default="dataset.json")
    parser.add_argument("--file_binary_json", type=str, default="dataset_binary.json")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Escenarios: nombre=ruta_train. Ej: control=../train synth25=../train_25")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Configurar escenarios
    if args.scenarios:
        scenarios = parse_scenarios(args.scenarios)
    else:
        scenarios = {"control": args.train_dir}

    # Guardar configuración
    config = vars(args).copy()
    config["scenarios_resolved"] = scenarios
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Ejecutar cada escenario ──
    scenario_data = {}  # {scenario_name: [run_data_0, ..., run_data_N]}

    for scenario_name, train_dir in scenarios.items():
        print(f"\n{'#'*60}")
        print(f"ESCENARIO: {scenario_name}  (train_dir={train_dir})")
        print(f"{'#'*60}")

        scenario_dir = os.path.join(args.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        all_run_data = []
        for i in range(args.n_runs):
            run_data = run_single(i, args, train_dir, scenario_dir)
            all_run_data.append(run_data)

        aggregate_results(all_run_data, scenario_dir)
        scenario_data[scenario_name] = all_run_data

    # ── Pruebas estadísticas entre escenarios ──
    run_statistical_tests(scenario_data, args.output_dir)

    print(f"\nExperimento completado. Resultados en: {args.output_dir}")


if __name__ == "__main__":
    main()
