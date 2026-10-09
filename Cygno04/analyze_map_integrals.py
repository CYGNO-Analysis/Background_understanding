import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Stile visivo ispirato a ROOT
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2

# -----------------------------------------------------------------------------
# 1. DIZIONARI DI CONFIGURAZIONE PER FILE
# -----------------------------------------------------------------------------
# Mappatura del file sul valore dell'asse X (es. Vdrift o VGEM)
RUN_MAP_X = {
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_000.root":   0,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_040.root":  40,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_140.root": 140,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_240.root": 240,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_340.root": 340,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_440.root": 440,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_540.root": 540,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_600.root": 600,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_740.root": 740,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_800.root": 800,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_840.root": 840,
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_940.root": 940,
}

# Range del fit 1D manuale per ciascun file: "nome_file": [fit_min, fit_max]
FIT_RANGES = {
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_000.root":[0,10],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_040.root":[10,15],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_140.root":[5,20],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_240.root":[7,20],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_340.root":[8,40],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_440.root":[5,40],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_540.root":[5,40],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_600.root":[5,40],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_740.root":[5,40],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_800.root":[5,40],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_840.root":[5,40],
    "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_940.root":[5,40],
}


def gaussian(x, a, mean, sigma):
    """Funzione Gaussiana per il fit."""
    return a * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def process_map_file(
    file_path,
    hist_name="prof2d_images",
    xmin=None, xmax=None,
    ymin=None, ymax=None,
    zmax=None,
    fit_range=None,
    output_dir="./plots_fit"
):
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
    # 1. TAGLI GEOMETRICI E TAGLIO Z_MAX SULLA MAPPA 2D
    # ---------------------------------------------------------
    x_mask = np.ones_like(x_centers, dtype=bool)
    if xmin is not None: x_mask &= (x_centers >= xmin)
    if xmax is not None: x_mask &= (x_centers <= xmax)

    y_mask = np.ones_like(y_centers, dtype=bool)
    if ymin is not None: y_mask &= (y_centers >= ymin)
    if ymax is not None: y_mask &= (y_centers <= ymax)

    cropped_data = data[np.ix_(y_mask, x_mask)]

    # Taglio dei bin non validi e azzeramento scariche (bincontent > zmax)
    valid_mask = (~np.isnan(cropped_data)) & (cropped_data > 0)
    if zmax is not None:
        valid_mask &= (cropped_data < zmax)

    clean_values = cropped_data[valid_mask]

    if len(clean_values) == 0:
        print(f"[WARNING] Nessun pixel valido trovato per {file_path}")
        return None, None

    # ---------------------------------------------------------
    # 2. COSTRUZIONE ISTOGRAMMA 1D
    # ---------------------------------------------------------
    nbins = 60
    counts, bin_edges = np.histogram(clean_values, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Errore statistico per bin (ROOT style: sqrt(N))
    stat_errors = np.sqrt(np.where(counts > 0, counts, 1.0))

    # ---------------------------------------------------------
    # 3. FIT GAUSSIANO NEL RANGE MANUALE
    # ---------------------------------------------------------
    if fit_range is not None:
        f_min, f_max = fit_range
    else:
        f_min, f_max = np.min(clean_values), np.max(clean_values)

    fit_mask = (bin_centers >= f_min) & (bin_centers <= f_max) & (counts > 0)
    x_fit = bin_centers[fit_mask]
    y_fit = counts[fit_mask]
    err_fit = stat_errors[fit_mask]

    fitted_mean = np.mean(clean_values)
    fitted_mean_err = np.std(clean_values) / np.sqrt(len(clean_values))

    popt, pcov = None, None
    if len(x_fit) >= 3:
        p0 = [np.max(y_fit), np.mean(x_fit), np.std(x_fit)]
        try:
            popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, sigma=err_fit, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))

            fitted_mean = popt[1]
            fitted_mean_err = perr[1]
        except Exception as e:
            print(f"[INFO] Fit fallito per {file_path}: {e}")

    # ---------------------------------------------------------
    # 4. PLOT 1D IN STILE ROOT (Punti + Errori Stat)
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot dei bin dell'istogramma come punti con barre d'errore sqrt(N)
    nonzero = counts > 0
    ax.errorbar(
        bin_centers[nonzero], counts[nonzero],
        yerr=stat_errors[nonzero], xerr=bin_width / 2.0,
        fmt='o', color='black', ecolor='black',
        markersize=3, elinewidth=1, capsize=0,
        label='Data (stat. err)'
    )

    # Disegno della curva di fit
    if popt is not None:
        x_curve = np.linspace(f_min, f_max, 200)
        y_curve = gaussian(x_curve, *popt)
        ax.plot(
            x_curve, y_curve, 'r-', lw=2,
            label=f'Gauss Fit\n$\mu$ = {popt[1]:.2f} $\pm$ {perr[1]:.2f}\n$\sigma$ = {abs(popt[2]):.2f}'
        )
        # Evidenzia la regione di fit
        ax.axvline(f_min, color='red', linestyle='--', alpha=0.5)
        ax.axvline(f_max, color='red', linestyle='--', alpha=0.5)

    base_name = os.path.basename(file_path).replace('.root', '')
    ax.set_title(f"Core Fit - {base_name}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Mean Amplitude / Pixel [ADC]", fontsize=11)
    ax.set_ylabel("Counts", fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fit_{base_name}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"fit_{base_name}.png"), bbox_inches='tight')
    plt.close(fig)

    return fitted_mean, fitted_mean_err


def main():
    parser = argparse.ArgumentParser(description="Calcolo e fit del core da mappe 2D TH2D.")
    parser.add_argument("--hist-name", type=str, default="prof2d_images")
    parser.add_argument("--xmin", type=float, default=None, help="Taglio min X [pixel]")
    parser.add_argument("--xmax", type=float, default=None, help="Taglio max X [pixel]")
    parser.add_argument("--ymin", type=float, default=None, help="Taglio min Y [pixel]")
    parser.add_argument("--ymax", type=float, default=None, help="Taglio max Y [pixel]")
    parser.add_argument("--zmax", type=float, default=None, help="Taglio max Z [bincontent] per rimuovere le scariche")
    parser.add_argument("--output-dir", type=str, default="./analysis_results")
    parser.add_argument("--x-title", type=str, default="V_{drift} [V]")

    args = parser.parse_args()

    x_vals = []
    y_vals = []
    y_errs = []

    print("--> Inizio analisi delle mappe 2D...")

    for file_path, x_val in RUN_MAP_X.items():
        print(f"\nElaborazione: {file_path} (X = {x_val})")

        fit_range = FIT_RANGES.get(file_path, None)

        mean_fit, err_fit = process_map_file(
            file_path=file_path,
            hist_name=args.hist_name,
            xmin=args.xmin, xmax=args.xmax,
            ymin=args.ymin, ymax=args.ymax,
            zmax=args.zmax,
            fit_range=fit_range,
            output_dir=os.path.join(args.output_dir, "fits")
        )

        if mean_fit is not None:
            x_vals.append(x_val)
            y_vals.append(mean_fit)
            y_errs.append(err_fit)

    if len(x_vals) == 0:
        print("[ERRORE] Nessun dato estratto.")
        return

    sorted_indices = np.argsort(x_vals)
    x_vals = np.array(x_vals)[sorted_indices]
    y_vals = np.array(y_vals)[sorted_indices]
    y_errs = np.array(y_errs)[sorted_indices]

    # ---------------------------------------------------------
    # GRAFICO FINALE: MEAN FITTATA VS PAROMETRO X
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        x_vals, y_vals, yerr=y_errs,
        fmt='o-', color='black', ecolor='crimson',
        markersize=5, elinewidth=1.2, capsize=2, capthick=1.2,
        label='Fitted Mean'
    )

    ax.set_title("Fitted Mean vs Run Parameter", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(args.x_title, fontsize=11)
    ax.set_ylabel("Fitted Mean Amplitude [ADC]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.legend()

    os.makedirs(args.output_dir, exist_ok=True)
    out_final_base = os.path.join(args.output_dir, "summary_fitted_mean_vs_X")
    plt.savefig(f"{out_final_base}.pdf", bbox_inches='tight')
    plt.savefig(f"{out_final_base}.png", bbox_inches='tight')
    plt.close(fig)

    print(f"\n=================================================")
    print(f"Analisi completata!")
    print(f"Grafici dei fit salvati in: {os.path.join(args.output_dir, 'fits')}")
    print(f"Grafico riassuntivo salvato in: {out_final_base}.pdf/.png")


if __name__ == "__main__":
    main()


