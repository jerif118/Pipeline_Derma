from pathlib import Path
from typing import Dict, List
from utils import (
    clear_dir, ensure_dir, proportional_count, safe_sample,
    copy_selected_images, build_dataset_json, summarize_selection, save_json
)

def generate_deficit_fill(
    real_by_class: Dict[str, List[Path]],
    synth_by_class: Dict[str, List[Path]],
    output_root: Path,
    proportions: List[float]
) -> None:
    mode_root = output_root / "train_deficit"
    clear_dir(mode_root)

    max_real = max(len(v) for v in real_by_class.values())

    print(f"\n{'='*60}")
    print(f" GENERANDO ESCENARIOS: DEFICIT FILL ")
    print(f"{'='*60}")

    for p in proportions:
        tag = f"train_{int(round(p * 100))}"
        scenario_dir = mode_root / tag
        ensure_dir(scenario_dir)
        
        print(f"\n[{tag}] Proporción configurada: {p*100:.0f}%")

        selected_real = {}
        selected_synth = {}
        target_per_class = {}

        for cls, real_files in real_by_class.items():
            n_real = len(real_files)
            deficit = max_real - n_real
            n_synth_needed = proportional_count(deficit, p)

            available_synth = synth_by_class.get(cls, [])
            selected_real[cls] = list(real_files)
            selected_synth[cls] = safe_sample(available_synth, n_synth_needed) if n_synth_needed > 0 else []
            target_per_class[cls] = n_real + n_synth_needed

        copy_selected_images(selected_real, selected_synth, scenario_dir)
        build_dataset_json(selected_real, selected_synth, scenario_dir)

        summary = summarize_selection(
            selected_real=selected_real,
            selected_synth=selected_synth,
            proportion_name=tag,
            mode_name="deficit_fill",
            target_per_class=target_per_class
        )
        summary["config"] = {
            "max_real_reference": max_real,
            "proportion": p
        }
        save_json(scenario_dir / "summary.json", summary)


def generate_total_mix(
    real_by_class: Dict[str, List[Path]],
    synth_by_class: Dict[str, List[Path]],
    output_root: Path,
    proportions: List[float]
) -> None:
    mode_root = output_root / "train_total"
    clear_dir(mode_root)

    print(f"\n{'='*60}")
    print(f" GENERANDO ESCENARIOS: TOTAL MIX ")
    print(f"{'='*60}")

    max_real = max(len(v) for v in real_by_class.values())

    for p in proportions:
        tag = f"train_{int(round(p * 100))}"
        scenario_dir = mode_root / tag
        ensure_dir(scenario_dir)
        
        print(f"\n[{tag}] Proporción configurada: {p*100:.0f}%")

        selected_real = {}
        selected_synth = {}
        target_per_class = {}

        for cls, real_files in real_by_class.items():
            target_total = max_real
            n_synth = proportional_count(target_total, p)
            n_real = target_total - n_synth

            available_real = real_files
            available_synth = synth_by_class.get(cls, [])

            actual_real = min(n_real, len(available_real))
            if actual_real < n_real:
                n_synth += (n_real - actual_real)

            selected_real[cls] = safe_sample(available_real, actual_real) if actual_real > 0 else []
            selected_synth[cls] = safe_sample(available_synth, n_synth) if n_synth > 0 else []
            target_per_class[cls] = len(selected_real[cls]) + len(selected_synth[cls])

        copy_selected_images(selected_real, selected_synth, scenario_dir)
        build_dataset_json(selected_real, selected_synth, scenario_dir)

        summary = summarize_selection(
            selected_real=selected_real,
            selected_synth=selected_synth,
            proportion_name=tag,
            mode_name="total_mix",
            target_per_class=target_per_class
        )
        summary["config"] = {
            "proportion": p,
            "target_mode": "max_real",
            "max_real_reference": max_real
        }
        save_json(scenario_dir / "summary.json", summary)