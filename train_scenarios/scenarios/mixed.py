from pathlib import Path

from utils import set_seed, load_real_images, load_synth_images, ensure_dir
from generators import generate_deficit_fill, generate_total_mix

REAL_TRAIN_DIR = r"../../train"
SYNTH_OK_DIR = r"../../synthetic_images"
OUTPUT_ROOT = r"../"

DEFICIT_PROPORTIONS = [0.25, 0.50, 0.75, 1.00]
TOTAL_MIX_PROPORTIONS = [0.25, 0.50, 0.75, 1.00]

SEED = 42

def main():
    set_seed(SEED)

    real_root = Path(REAL_TRAIN_DIR)
    synth_root = Path(SYNTH_OK_DIR)
    output_root = Path(OUTPUT_ROOT)

    try:
        real_by_class = load_real_images(real_root)
        synth_by_class = load_synth_images(synth_root)
    except FileNotFoundError as e:
        print(f"Advertencia: {e}")
        return

    ensure_dir(output_root)

    print("Generando Deficit Fill...")
    generate_deficit_fill(
        real_by_class=real_by_class,
        synth_by_class=synth_by_class,
        output_root=output_root,
        proportions=DEFICIT_PROPORTIONS
    )

    print("Generando Total Mix...")
    generate_total_mix(
        real_by_class=real_by_class,
        synth_by_class=synth_by_class,
        output_root=output_root,
        proportions=TOTAL_MIX_PROPORTIONS
    )

    print("Todos los escenarios han sido generados exitosamente.")

if __name__ == "__main__":
    main()
