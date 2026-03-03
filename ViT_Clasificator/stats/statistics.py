import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from itertools import combinations

from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
from statsmodels.stats.contingency_tables import cochrans_q as _sm_cochrans_q
from statsmodels.stats.multitest import multipletests


ALPHA = 0.05

def shapiro_wilk(values: np.ndarray) -> dict:
    if len(values) < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "is_normal": False}
    stat, p = sp_stats.shapiro(values)
    return {"statistic": float(stat), "p_value": float(p), "is_normal": p >= ALPHA}


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Las muestras pareadas deben tener la misma forma: {a.shape} vs {b.shape}")

    diffs = a - b
    if np.allclose(diffs, 0.0):
        return {
            "test": "no_difference",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "normality": {"statistic": float("nan"), "p_value": float("nan"), "is_normal": True},
            "note": "all_paired_differences_are_zero",
        }

    normality = shapiro_wilk(diffs)

    if normality["is_normal"]:
        stat, p = sp_stats.ttest_rel(a, b)
        test_name = "paired_t_test"
    else:
        try:
            stat, p = sp_stats.wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
            test_name = "wilcoxon_signed_rank"
        except ValueError:
            return {
                "test": "no_difference",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "normality": normality,
                "note": "wilcoxon_not_defined_for_input",
            }

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < ALPHA,
        "normality": normality,
    }


def cochrans_q(correct_matrix: np.ndarray) -> dict:
    n, k = correct_matrix.shape
    if k < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}

    q_result = _sm_cochrans_q(correct_matrix)

    if hasattr(q_result, "statistic"):
        stat = q_result.statistic
        p_val = q_result.pvalue
        df = getattr(q_result, "df", k - 1)
    else:
        stat, p_val, df = q_result

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "df": int(df),
        "significant": p_val < ALPHA,
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    table = np.array([
        [((correct_a == 1) & (correct_b == 1)).sum(),
         ((correct_a == 1) & (correct_b == 0)).sum()],
        [((correct_a == 0) & (correct_b == 1)).sum(),
         ((correct_a == 0) & (correct_b == 0)).sum()],
    ])

    result = _sm_mcnemar(table, exact=False, correction=True)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": result.pvalue < ALPHA,
    }


def mcnemar_pairwise(correct_matrix: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    k = correct_matrix.shape[1]
    if labels is None:
        labels = [f"model_{i}" for i in range(k)]

    rows = []
    for i, j in combinations(range(k), 2):
        res = mcnemar_test(correct_matrix[:, i], correct_matrix[:, j])
        rows.append({
            "model_a": labels[i],
            "model_b": labels[j],
            "statistic": res["statistic"],
            "p_value": res["p_value"],
        })

    return pd.DataFrame(rows)


def holm_bonferroni(p_values: np.ndarray) -> dict:
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=ALPHA, method="holm")

    return {
        "p_values_adjusted": p_adjusted.tolist(),
        "rejected": rejected.tolist(),
        "alpha": ALPHA,
    }


def mcnemar_posthoc_corrected(correct_matrix: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    df = mcnemar_pairwise(correct_matrix, labels)
    if len(df) == 0:
        return df
    correction = holm_bonferroni(df["p_value"].values)
    df["p_adjusted"] = correction["p_values_adjusted"]
    df["significant"] = correction["rejected"]
    return df


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_pred: np.ndarray,
                 n_bootstrap: int = 10000, ci: float = 0.95,
                 seed: int = 42, **metric_kwargs) -> dict:
    def statistic(idx):
        idx = idx.astype(int)
        return metric_fn(y_true[idx], y_pred[idx], **metric_kwargs)

    rng = np.random.default_rng(seed)
    indices = (np.arange(len(y_true)),)
    result = sp_stats.bootstrap(
        indices, statistic, n_resamples=n_bootstrap,
        confidence_level=ci, random_state=rng, method="percentile",
    )

    return {
        "ci_lower": float(result.confidence_interval.low),
        "ci_upper": float(result.confidence_interval.high),
        "ci_level": ci,
        "se": float(result.standard_error),
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_ci_multiclass(y_true, y_pred, y_proba, n_bootstrap=10000, ci=0.95, seed=42):
    from sklearn.metrics import (
        balanced_accuracy_score,
        roc_auc_score,
        f1_score,
        recall_score,
    )

    n_labels = y_proba.shape[1]
    labels_list = list(range(n_labels))
    rng = np.random.default_rng(seed)

    def _make_statistic(fn_name):
        def statistic(idx):
            idx = idx.astype(int)
            yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
            if fn_name == "balanced_accuracy":
                return balanced_accuracy_score(yt, yp)
            elif fn_name == "f1_macro":
                return f1_score(yt, yp, average="macro", zero_division=0)
            elif fn_name == "f1_weighted":
                return f1_score(yt, yp, average="weighted", zero_division=0)
            elif fn_name == "recall_macro":
                return recall_score(yt, yp, average="macro", zero_division=0)
            elif fn_name == "recall_weighted":
                return recall_score(yt, yp, average="weighted", zero_division=0)
            elif fn_name == "auc_roc_ovr":
                try:
                    return roc_auc_score(yt, ypr, multi_class="ovr",
                                         average="macro", labels=labels_list)
                except ValueError:
                    return float("nan")
        return statistic

    metric_names = ["balanced_accuracy", "auc_roc_ovr", "f1_macro", "f1_weighted",
                    "recall_macro", "recall_weighted"]
    indices = (np.arange(len(y_true)),)

    result = {}
    for m in metric_names:
        boot = sp_stats.bootstrap(
            indices, _make_statistic(m), n_resamples=n_bootstrap,
            confidence_level=ci, random_state=np.random.default_rng(seed),
            method="percentile",
        )
        result[m] = {
            "ci_lower": float(boot.confidence_interval.low),
            "ci_upper": float(boot.confidence_interval.high),
            "se": float(boot.standard_error),
            "ci_level": ci,
        }
    return result


def scalar_bootstrap_ci(values, ci: float = 0.95, n_bootstrap: int = 10000, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    def statistic(idx):
        idx = idx.astype(int)
        return np.mean(values[idx])

    indices = (np.arange(len(values)),)
    result = sp_stats.bootstrap(
        indices, statistic, n_resamples=n_bootstrap,
        confidence_level=ci, random_state=rng, method="percentile",
    )
    return {
        "ci_lower": float(result.confidence_interval.low),
        "ci_upper": float(result.confidence_interval.high),
        "se": float(result.standard_error),
        "ci_level": ci,
        "n_runs": len(values),
    }


def compare_scenarios(metrics_per_scenario: dict[str, np.ndarray]) -> dict:
    names = list(metrics_per_scenario.keys())
    base_name = names[0]
    base_values = metrics_per_scenario[base_name]

    comparisons = {}
    for name in names[1:]:
        comparisons[f"{base_name}_vs_{name}"] = paired_comparison(base_values, metrics_per_scenario[name])

    return comparisons
