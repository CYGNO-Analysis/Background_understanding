import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt

# Stile visivo
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2


def create_veto_mask(
    input_root_file,
    hist_name="prof2d_images",
    threshold_down=None,
    threshold_up=None,
    output_dir="./veto_masks"
):
    """
    Legge un TH2D da file ROOT, individua i pixel con contenuto < threshold_down
    oppure > threshold_up, salva la maschera in formato .npz e genera il plot della mappa di veto.
    """
    if not os.path.exists(input_root_file):
        raise FileNotFoundError(f"File non trovato: {input_root_file}")

    print(f"--> Caricamento {hist_name} da: {input_root_file}")
    with uproot.open(input_root_file) as f:
        if hist_name not in f:
            raise KeyError(f"Istogramma '{hist_name}' non trovato nel file!")

        hist = f[hist_name]
        vals, xedges, yedges = hist.to_numpy()

    # Trasposta per allineare (nx, ny) di Uproot a (ny, nx) di Matplotlib/NumPy
    data = vals.T

    # 1. Creazione della Maschera Booleana di VETO
    # Partiamo considerando vetati i valori NaN / invalidi
    veto_mask = np.isnan(data)

    # Applichiamo la soglia inferiore se definita
    if threshold_down is not None:
        veto_mask |= (data < threshold_down)

    # Applichiamo la soglia superiore se definita
    if threshold_up is not None:
        veto_mask |= (data > threshold_up)

    n_vetoed = np.sum(veto_mask)
    total_pixels = veto_mask.size
    print(f"Pixel da vetare: {n_vetoed} / {total_pixels} ({n_vetoed / total_pixels * 100:.2f}%)")
    print(f"  > Criteria: Low threshold = {threshold_down}, High threshold = {threshold_up}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_root_file).replace(".root", "")

    # 2. Salvataggio della Maschera (.npz)
    mask_file_path = os.path.join(output_dir, f"veto_mask_{base_name}.npz")
    np.savez_compressed(
        mask_file_path,
        veto_mask=veto_mask,
        xedges=xedges,
        yedges=yedges,
        threshold_down=threshold_down if threshold_down is not None else np.nan,
        threshold_up=threshold_up if threshold_up is not None else np.nan
    )
    print(f"Maschera salvata in: {mask_file_path}")

    # 3. Plot della Mappa del Veto
    fig, ax = plt.subplots(figsize=(7, 10))

    # 1 (Rosso) = Vetato, NaN (Trasparente) = OK
    display_map = np.where(veto_mask, 1.0, np.nan)

    mesh = ax.pcolormesh(
        xedges, yedges, display_map,
        shading='flat',
        cmap='Reds',
        vmin=0, vmax=1
    )

    title_str = f"Veto Pixel Map\n{base_name}\n"
    conds = []
    if threshold_down is not None: conds.append(f"< {threshold_down}")
    if threshold_up is not None: conds.append(f"> {threshold_up}")
    title_str += f"(Cut: {' or '.join(conds)})" if conds else "(Only NaNs)"

    ax.set_title(title_str, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("X [pixel]", fontsize=12)
    ax.set_ylabel("Y [pixel]", fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    plot_path_base = os.path.join(output_dir, f"plot_veto_{base_name}")
    plt.savefig(f"{plot_path_base}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{plot_path_base}.png", format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Plot del veto salvati in: {plot_path_base}.pdf/.png")
    return mask_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera e salva la maschera di veto per pixel < threshold_down o > threshold_up.")
    parser.add_argument("--input", type=str, required=True, help="Path del file ROOT di input")
    parser.add_argument("--hist-name", type=str, default="prof2d_images", help="Nome dell'istogramma 2D (default: prof2d_images)")
    parser.add_argument("--threshold-down", type=float, default=None, help="Soglia inferiore per vetare i pixel (valori < threshold_down)")
    parser.add_argument("--threshold-up", type=float, default=None, help="Soglia superiore per vetare i pixel (valori > threshold_up)")
    parser.add_argument("--output-dir", type=str, default="./veto_masks", help="Directory dove salvare maschera e plot")

    args = parser.parse_args()

    create_veto_mask(
        input_root_file=args.input,
        hist_name=args.hist_name,
        threshold_down=args.threshold_down,
        threshold_up=args.threshold_up,
        output_dir=args.output_dir
    )
