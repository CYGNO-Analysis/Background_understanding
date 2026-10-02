import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_2d_maps(
    input_root_file,
    output_dir="./plots",
    use_log=False,
    zmin_prof=None, zmax_prof=None,
    zmin_int=None, zmax_int=None,
    zmin_occ=None, zmax_occ=None
):
    with uproot.open(input_root_file) as f:
        occ2d_hist = f["occ2d_images"]
        int2d_hist = f["int2d_images"]
        prof2d_hist = f["prof2d_images"]

        occ_vals, xedges, yedges = occ2d_hist.to_numpy()
        int_vals, _, _ = int2d_hist.to_numpy()
        prof_vals, _, _ = prof2d_hist.to_numpy()

    # Trasposta per allineare (nx, ny) di Uproot a (ny, nx) di Matplotlib
    occ_vals = occ_vals.T
    int_vals = int_vals.T
    prof_vals = prof_vals.T

    # Masking dei bin vuoti (occupancy = 0) -> diventano trasparenti/bianchi
    zero_mask = (occ_vals == 0)

    occ_masked = np.where(zero_mask, np.nan, occ_vals)
    int_masked = np.where(zero_mask, np.nan, int_vals)
    prof_masked = np.where(zero_mask, np.nan, prof_vals)

    # Nota: puoi usare 'rainbow', 'jet', oppure 'turbo'
    COLORMAP = "rainbow" 

    maps_to_plot = {
        "occ2d_images": {
            "data": occ_masked,
            "title": "Pixel Occupancy Map",
            "zlabel": "Hits / Bin",
            "zmin": zmin_occ,
            "zmax": zmax_occ,
        },
        "int2d_images": {
            "data": int_masked,
            "title": "Integrated Charge Map",
            "zlabel": "Total Charge [ADC]",
            "zmin": zmin_int,
            "zmax": zmax_int,
        },
        "prof2d_images": {
            "data": prof_masked,
            "title": "Mean Amplitude Profile 2D",
            "zlabel": "Mean Charge / Hit [ADC]",
            "zmin": zmin_prof,
            "zmax": zmax_prof,
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    for name, cfg in maps_to_plot.items():
        # Impostiamo una proporzione della figura (figsize) più stretta e alta
        # per rispecchiare la geometria delle 3 camere (altezza total_height >> larghezza width)
        fig, ax = plt.subplots(figsize=(7, 10))

        norm = None
        if use_log:
            vmin = cfg["zmin"] if (cfg["zmin"] is not None and cfg["zmin"] > 0) else np.nanmin(cfg["data"])
            vmax = cfg["zmax"] if cfg["zmax"] is not None else np.nanmax(cfg["data"])
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = cfg["zmin"]
            vmax = cfg["zmax"]

        mesh = ax.pcolormesh(
            xedges, yedges, cfg["data"],
            shading='flat',
            cmap=COLORMAP,
            vmin=vmin if not use_log else None,
            vmax=vmax if not use_log else None,
            norm=norm
        )

        cbar = fig.colorbar(mesh, ax=ax, pad=0.03, fraction=0.046)
        cbar.set_label(cfg["zlabel"], fontsize=12)

        ax.set_title(cfg["title"], fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("X [pixel]", fontsize=12)
        ax.set_ylabel("Y [pixel]", fontsize=12)
        
        # Mantiene le proporzioni geometriche reali dei pixel
        ax.set_aspect('equal')
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")

        # Salva in formato PDF rimuovendo tutto lo spazio bianco superfluo attorno
        output_path = os.path.join(output_dir, name)
        
        # bbox_inches='tight' ELIMINA I MARGINI BIANCHI AI LATI E CENTRA IL PLOT NEL PDF
        plt.savefig(f"{output_path}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
        plt.savefig(f"{output_path}.png", format="png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        print(f"PDF/PNG saved in: {output_path}.pdf/png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D delle mappe CYGNO saltando i bin vuoti.")
    parser.add_argument("--input", type=str, required=True, help="Path del file ROOT di input")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Directory di output per le immagini")
    parser.add_argument("--log", action="store_true", help="Usa scala logaritmica per l'asse Z")

    # Range asse Z per il Profilo 2D
    parser.add_argument("--zmin-prof", type=float, default=None, help="Z min per Profilo 2D")
    parser.add_argument("--zmax-prof", type=float, default=None, help="Z max per Profilo 2D")

    # Range asse Z per l'Intensità
    parser.add_argument("--zmin-int", type=float, default=None, help="Z min per Integrazione 2D")
    parser.add_argument("--zmax-int", type=float, default=None, help="Z max per Integrazione 2D")

    # Range asse Z per l'Occupancy
    parser.add_argument("--zmin-occ", type=float, default=None, help="Z min per Occupancy 2D")
    parser.add_argument("--zmax-occ", type=float, default=None, help="Z max per Occupancy 2D")

    args = parser.parse_args()

    plot_2d_maps(
        input_root_file=args.input,
        output_dir=args.output_dir,
        use_log=args.log,
        zmin_prof=args.zmin_prof, zmax_prof=args.zmax_prof,
        zmin_int=args.zmin_int, zmax_int=args.zmax_int,
        zmin_occ=args.zmin_occ, zmax_occ=args.zmax_occ
    )
