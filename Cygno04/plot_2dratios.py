import os
import argparse
import numpy as np
import uproot
import ROOT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Applicazione stile grafico pulito (simile a ROOT)
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2

# -----------------------------------------------------------------------------
# NOMINE DEI FILE REALI
# -----------------------------------------------------------------------------
INPUT_FILES = {
    (440, 600): "cygno04_sideA_exp7s_nostdcut_VGEM_440_VD_600.root",
    (440, 800): "cygno04_sideA_exp7s_nostdcut_VGEM_440_VD_800.root",
    (440, 940): "cygno04_sideA_exp7s_nostdcut_VGEM_440_VD_940.root",
    (450, 600): "cygno04_sideA_exp7s_nostdcut_VGEM_450_VD_600.root",
    (450, 800): "cygno04_sideA_exp7s_nostdcut_VGEM_450_VD_800.root",
    (450, 940): "cygno04_sideA_exp7s_nostdcut_VGEM_450_VD_940.root",
    (460, 600): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_600.root",
    (460, 800): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_800.root",
    (460, 940): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_940.root",

    # (440, 600): "cygno04_sideA_muons_VGEM_440_VD_600.root",
    # (440, 800): "cygno04_sideA_muons_VGEM_440_VD_800.root",
    # (440, 940): "cygno04_sideA_muons_VGEM_440_VD_940.root",
    # (450, 600): "cygno04_sideA_muons_VGEM_450_VD_600.root",
    # (450, 800): "cygno04_sideA_muons_VGEM_450_VD_800.root",
    # (450, 940): "cygno04_sideA_muons_VGEM_450_VD_940.root",
    # (460, 600): "cygno04_sideA_muons_VGEM_460_VD_600.root",
    # (460, 800): "cygno04_sideA_muons_VGEM_460_VD_800.root",
    # (460, 940): "cygno04_sideA_muons_VGEM_460_VD_940.root",
}


def load_profile_map(file_path, hist_name="prof2d_images"):
    """Legge la mappa 2D da un file ROOT e restituisce la matrice e i bordi degli assi."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")

    with uproot.open(file_path) as f:
        hist = f[hist_name]
        vals, xedges, yedges = hist.to_numpy()

    # Trasposta per allineare (nx, ny) di Uproot a (ny, nx) di Matplotlib
    return vals.T, xedges, yedges


def plot_and_save_ratio(ratio_matrix, xedges, yedges, title, out_base_path, zmin=None, zmax=None, use_log=False):
    """Genera e salva la mappa 2D del rapporto."""
    fig, ax = plt.subplots(figsize=(7, 10))

    norm = None
    if use_log:
        vmin = zmin if (zmin is not None and zmin > 0) else np.nanmin(ratio_matrix)
        vmax = zmax if zmax is not None else np.nanmax(ratio_matrix)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = zmin
        vmax = zmax

    mesh = ax.pcolormesh(
        xedges, yedges, ratio_matrix,
        shading='flat',
        cmap="rainbow",
        vmin=vmin if not use_log else None,
        vmax=vmax if not use_log else None,
        norm=norm
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_label("Mean Amplitude Ratio", fontsize=12)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("X [pixel]", fontsize=12)
    ax.set_ylabel("Y [pixel]", fontsize=12)

    ax.set_aspect('equal')
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    plt.savefig(f"{out_base_path}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{out_base_path}.png", format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_and_save_single_projection(
    centers, mean_vals, err_vals, axis_label, title, out_file_path, ymin=None, ymax=None
):
    """
    Genera un plot 1D in stile ROOT (punti con barre d'errore sulla media).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot con stile ROOT (punti e barre d'errore)
    ax.errorbar(
        centers, mean_vals, yerr=err_vals,
        fmt='o', color='black', ecolor='black',
        markersize=3, elinewidth=1, capsize=1.5,
        label='Mean ± SEM'
    )

    ax.set_title(f"{axis_label} Projection - {title}", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(f"{axis_label} [pixel]", fontsize=12)
    ax.set_ylabel("Mean Ratio", fontsize=12)

    # Impostazione del range dell'asse Y (corrispondente a zmin/zmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    ax.grid(True, linestyle="--", alpha=0.4, color="gray")

    plt.tight_layout()
    plt.savefig(f"{out_file_path}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{out_file_path}.png", format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_and_save_1d_projections(ratio_matrix, xedges, yedges, title, out_base_path, zmin=None, zmax=None):
    """
    Calcola la media e l'errore sulla media (SEM = std / sqrt(N)) lungo X e Y
    e salva separatamente i plot 1D in stile ROOT.
    """
    x_centers = (xedges[:-1] + xedges[1:]) / 2.0
    y_centers = (yedges[:-1] + yedges[1:]) / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        # --- PROIEZIONE SU X (Media lungo Y) ---
        proj_x_mean = np.nanmean(ratio_matrix, axis=0)
        proj_x_std = np.nanstd(ratio_matrix, axis=0)
        counts_x = np.sum(~np.isnan(ratio_matrix), axis=0)
        proj_x_sem = np.where(counts_x > 0, proj_x_std / np.sqrt(counts_x), np.nan)

        # --- PROIEZIONE SU Y (Media lungo X) ---
        proj_y_mean = np.nanmean(ratio_matrix, axis=1)
        proj_y_std = np.nanstd(ratio_matrix, axis=1)
        counts_y = np.sum(~np.isnan(ratio_matrix), axis=1)
        proj_y_sem = np.where(counts_y > 0, proj_y_std / np.sqrt(counts_y), np.nan)

    # 1. Plot separato per X
    plot_and_save_single_projection(
        centers=x_centers,
        mean_vals=proj_x_mean,
        err_vals=proj_x_sem,
        axis_label="X",
        title=title,
        out_file_path=f"{out_base_path}_1Dproj_X",
        ymin=zmin,
        ymax=zmax
    )

    # 2. Plot separato per Y
    plot_and_save_single_projection(
        centers=y_centers,
        mean_vals=proj_y_mean,
        err_vals=proj_y_sem,
        axis_label="Y",
        title=title,
        out_file_path=f"{out_base_path}_1Dproj_Y",
        ymin=zmin,
        ymax=zmax
    )


def compute_all_ratios(input_dir="./", output_root="./profile_ratios.root", output_plot_dir="./ratio_plots", use_log=False):
    os.makedirs(output_plot_dir, exist_ok=True)

    print("--> Caricamento delle mappe di profilo 2D...")

    def get_path(vgem, vdrift):
        return os.path.join(input_dir, INPUT_FILES[(vgem, vdrift)])

    # Mappe per Vdrift = 940 V
    map_g460_d940, xedges, yedges = load_profile_map(get_path(460, 940))
    map_g450_d940, _, _          = load_profile_map(get_path(450, 940))
    map_g440_d940, _, _          = load_profile_map(get_path(440, 940))

    # Mappe per VGEM = 460 V
    map_g460_d800, _, _          = load_profile_map(get_path(460, 800))
    map_g460_d600, _, _          = load_profile_map(get_path(460, 600))

    # -------------------------------------------------------------------------
    # DIZIONARIO DEFINIZIONE RAPPORTI CON zmin E zmax DEDICATI
    # Modifica zmin e zmax per ciascun rapporto in base alle tue esigenze!
    # -------------------------------------------------------------------------
    ratios_definition = {
        "ratio_vdrift940_vgem460_vs_450": {
            "num": map_g460_d940, "den": map_g450_d940,
            "title": "Profile Ratio: VGEM 460V / 450V (Vdrift = 940V)",
            "filename": "prof2d_ratio_images_Vdrift940_VGEM460_vs_450",
            "zmin": 1.0, "zmax": 2.0
        },
        "ratio_vdrift940_vgem460_vs_440": {
            "num": map_g460_d940, "den": map_g440_d940,
            "title": "Profile Ratio: VGEM 460V / 440V (Vdrift = 940V)",
            "filename": "prof2d_ratio_images_Vdrift940_VGEM460_vs_440",
            "zmin": 1.0, "zmax": 3.0
        },
        "ratio_vgem460_vdrift940_vs_800": {
            "num": map_g460_d940, "den": map_g460_d800,
            "title": "Profile Ratio: Vdrift 940V / 800V (VGEM = 460V)",
            "filename": "prof2d_ratio_images_VGEM460_Vdrift940_vs_800",
            "zmin": 1.0, "zmax": 1.5
        },
        "ratio_vgem460_vdrift940_vs_600": {
            "num": map_g460_d940, "den": map_g460_d600,
            "title": "Profile Ratio: Vdrift 940V / 600V (VGEM = 460V)",
            "filename": "prof2d_ratio_images_VGEM460_Vdrift940_vs_600",
            "zmin": 1.0, "zmax": 2.0
        },
    }

    tfout = ROOT.TFile(output_root, "recreate")

    for key, cfg in ratios_definition.items():
        print(f"\nCalcolo rapporto: {cfg['title']}")

        num = cfg["num"]
        den = cfg["den"]

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where((den > 0) & (~np.isnan(num)) & (~np.isnan(den)), num / den, np.nan)

        out_plot_path = os.path.join(output_plot_dir, cfg["filename"])
        
        # Estraggo zmin e zmax specifici per questo rapporto
        rzmin = cfg.get("zmin", None)
        rzmax = cfg.get("zmax", None)

        # 1. Plot Mappa 2D
        plot_and_save_ratio(
            ratio, xedges, yedges, cfg["title"], out_plot_path, zmin=rzmin, zmax=rzmax, use_log=use_log
        )

        # 2. Plot Proiezioni 1D Separati (Stile ROOT: Punti + Errori)
        plot_and_save_1d_projections(
            ratio, xedges, yedges, cfg["title"], out_plot_path, zmin=rzmin, zmax=rzmax
        )

        # 3. Salvataggio in File ROOT
        ratio_root = np.nan_to_num(ratio, nan=0.0).T
        h2_ratio = ROOT.TH2D(key, cfg["title"], len(xedges)-1, xedges[0], xedges[-1], len(yedges)-1, yedges[0], yedges[-1])

        x_idx, y_idx = np.where(ratio_root > 0)
        x_coords = (xedges[x_idx] + xedges[x_idx + 1]) / 2.0
        y_coords = (yedges[y_idx] + yedges[y_idx + 1]) / 2.0
        weights = ratio_root[x_idx, y_idx].astype(np.float64)

        if len(x_coords) > 0:
            h2_ratio.FillN(len(x_coords), x_coords, y_coords, weights)

        tfout.cd()
        h2_ratio.Write()

    tfout.Close()
    print(f"\n==========================================")
    print(f"Salvataggio completato in {output_root} e nella directory {output_plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola i rapporti delle mappe 2D e i profili delle proiezioni 1D su X e Y.")
    parser.add_argument("--input-dir", type=str, default="./", help="Directory contenente i file .root di input")
    parser.add_argument("--output-root", type=str, default="./profile_ratios.root", help="Path del file ROOT di output")
    parser.add_argument("--output-dir", type=str, default="./ratio_plots", help="Directory per i plot PDF/PNG")
    parser.add_argument("--log", action="store_true", help="Usa scala logaritmica per l'asse Z/Y")

    args = parser.parse_args()

    compute_all_ratios(
        input_dir=args.input_dir,
        output_root=args.output_root,
        output_plot_dir=args.output_dir,
        use_log=args.log
    )
