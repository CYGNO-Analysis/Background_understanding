import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt

# Stile visivo per i grafici (stile ROOT / Physics)
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2

# -----------------------------------------------------------------------------
# 1. DIZIONARIO MAPPATURA FILE -> VALORE X (es. Vdrift o VGEM)
# -----------------------------------------------------------------------------
RUN_MAP_X = {
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_000.root":   0,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_040.root":  40,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_140.root": 140,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_240.root": 240,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_340.root": 340,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_440.root": 440,
    #"cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_540.root": 540,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_640.root": 640,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_740.root": 740,
    #"cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_800.root": 800,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_840.root": 840,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_940.root": 940,
}

def process_map_file(
    file_path,
    hist_name="prof2d_images",
    xmin=None, xmax=None,
    ymin=None, ymax=None,
    zmax=None,
    veto_mask_path=None
):
    """
    Legge la mappa 2D (prof2d_images), applica i tagli geometrici, il taglio zmax (scariche)
    e la maschera di veto .npz facoltativa, quindi calcola la somma dei pixel divisa per N.
    """
    if not os.path.exists(file_path):
        print(f"[WARNING] File non trovato: {file_path}")
        return None, None

    with uproot.open(file_path) as f:
        if hist_name not in f:
            print(f"[WARNING] Istogramma '{hist_name}' non trovato in {file_path}")
            return None, None

        hist = f[hist_name]
        vals, xedges, yedges = hist.to_numpy()

    # Trasposta per allineare gli assi Uproot (nx, ny) -> (ny, nx)
    data = vals.T

    x_centers = (xedges[:-1] + xedges[1:]) / 2.0
    y_centers = (yedges[:-1] + yedges[1:]) / 2.0

    # ---------------------------------------------------------
    # 1. SELEZIONE REGIONE D'INTERESSE (ROI GEOMETRICA)
    # ---------------------------------------------------------
    x_mask = np.ones_like(x_centers, dtype=bool)
    if xmin is not None: x_mask &= (x_centers >= xmin)
    if xmax is not None: x_mask &= (x_centers <= xmax)

    y_mask = np.ones_like(y_centers, dtype=bool)
    if ymin is not None: y_mask &= (y_centers >= ymin)
    if ymax is not None: y_mask &= (y_centers <= ymax)

    cropped_data = data[np.ix_(y_mask, x_mask)].copy()

    # ---------------------------------------------------------
    # 2. APPLICAZIONE MASCHERA DI VETO (SE PRESENTE)
    # ---------------------------------------------------------
    if veto_mask_path and os.path.exists(veto_mask_path):
        veto_data = np.load(veto_mask_path)
        external_veto = veto_data["veto_mask"]
        # Ritagliamo la maschera di veto sulla stessa ROI
        veto_roi = external_veto[np.ix_(y_mask, x_mask)]
        cropped_data[veto_roi] = np.nan

    # ---------------------------------------------------------
    # 3. TAGLI SU ZMAX (SCARICHE) E VALORI INVALIDI
    # ---------------------------------------------------------
    valid_mask = (~np.isnan(cropped_data)) & (cropped_data > 0)
    if zmax is not None:
        valid_mask &= (cropped_data < zmax)

    clean_values = cropped_data[valid_mask]

    n_pixels = len(clean_values)
    if n_pixels == 0:
        print(f"[WARNING] Nessun pixel valido trovato per {file_path}")
        return None, None

    # ---------------------------------------------------------
    # 4. CALCOLO SOMMA / N_PIXELS E ERRORE SULLA MEDIA
    # ---------------------------------------------------------
    total_sum = np.sum(clean_values)
    mean_val = total_sum / float(n_pixels)
    
    # Errore Standard sulla Media: std / sqrt(N)
    std_val = np.std(clean_values)
    sem_val = std_val / np.sqrt(n_pixels)

    print(f"  > Pixel usati: {n_pixels} | Somma: {total_sum:.2f} | Somma/N: {mean_val:.3f} ± {sem_val:.3f}")

    return mean_val, sem_val


def main():
    parser = argparse.ArgumentParser(description="Calcolo della media pura (Somma Z / N_pixel) da mappe 2D TH2D.")
    parser.add_argument("--hist-name", type=str, default="prof2d_images", help="Nome dell'istogramma TH2D")
    parser.add_argument("--xmin", type=float, default=None, help="Taglio min X [pixel]")
    parser.add_argument("--xmax", type=float, default=None, help="Taglio max X [pixel]")
    parser.add_argument("--ymin", type=float, default=None, help="Taglio min Y [pixel]")
    parser.add_argument("--ymax", type=float, default=None, help="Taglio max Y [pixel]")
    parser.add_argument("--zmax", type=float, default=None, help="Taglio max Z [bincontent] per rimuovere le scariche")
    parser.add_argument("--veto-mask", type=str, default=None, help="Path del file .npz contenente la maschera di veto")
    parser.add_argument("--output-dir", type=str, default="./analysis_results", help="Directory per i risultati")
    parser.add_argument("--x-title", type=str, default="V_{drift} [V]", help="Etichetta dell'asse X")

    args = parser.parse_args()

    x_vals = []
    y_vals = []
    y_errs = []

    print("--> Inizio analisi delle mappe 2D (Somma / N_pixel)...")

    for file_path, x_val in RUN_MAP_X.items():
        print(f"\nElaborazione: {file_path} (X = {x_val})")

        mean_val, err_val = process_map_file(
            file_path=file_path,
            hist_name=args.hist_name,
            xmin=args.xmin, xmax=args.xmax,
            ymin=args.ymin, ymax=args.ymax,
            zmax=args.zmax,
            veto_mask_path=args.veto_mask
        )

        if mean_val is not None:
            x_vals.append(x_val)
            y_vals.append(mean_val)
            y_errs.append(err_val)

    if len(x_vals) == 0:
        print("[ERRORE] Nessun dato estratto. Verificare i file di input.")
        return

    # Ordinamento dei dati lungo l'asse X
    sorted_indices = np.argsort(x_vals)
    x_vals = np.array(x_vals)[sorted_indices]
    y_vals = np.array(y_vals)[sorted_indices]
    y_errs = np.array(y_errs)[sorted_indices]

    # ---------------------------------------------------------
    # GRAFICO FINALE: (SOMMA / N) VS PAROMETRO X (es. Vdrift)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        x_vals, y_vals, yerr=y_errs,
        fmt='o-', color='black', ecolor='crimson',
        markersize=5, elinewidth=1.2, capsize=3, capthick=1.2,
        label=r'$\sum Z_{i} / N_{\mathrm{pixel}}$'
    )

    ax.set_title("Mean Pixel Value vs Run Parameter", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(args.x_title, fontsize=12)
    ax.set_ylabel("Mean Amplitude / Pixel [ADC]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.legend(loc='best', frameon=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_final_base = os.path.join(args.output_dir, "summary_mean_sum_div_N_vs_X")
    plt.savefig(f"{out_final_base}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{out_final_base}.png", format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"\n=================================================")
    print(f"Analisi completata!")
    print(f"Grafico riassuntivo salvato in: {out_final_base}.pdf e .png")
    print(f"=================================================")


if __name__ == "__main__":
    main()


