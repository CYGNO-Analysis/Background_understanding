import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import cygno as cy
import swiftlib as sw

def create_2dmaps_from_midas(
    run_start,
    run_end,
    output_pdf_dir="./plots_midas",
    tag="LNGS",
    tmpdir="/tmp/",
    width=4096,
    height_single_cam=2304,
    num_cameras=3,
    cimax=65535,         # Threshold massimo per escludere saturazione/artefatti
    pedestal_file=None   # Opzionale: array o percorso se si vuole sottrare il piedistallo
):
    total_height = height_single_cam * num_cameras

    # Accumulatori 2D (y, x) per l'intero detector
    # Usiamo float64 per evitare qualsiasi problema di overflow sum
    charge_sum = np.zeros((total_height, width), dtype=np.float64)
    occupancy_sum = np.zeros((total_height, width), dtype=np.float64)

    for run_num in range(run_start, run_end + 1):
        print(f"\n---> Scaricamento/Apertura MIDAS per Run: {run_num:05d}")
        try:
            mf = sw.swift_download_midas_file(run_num, tmpdir, tag)
        except Exception as e:
            print(f"ERRORE nel recupero del file MIDAS per il run {run_num}: {e}")
            continue

        event_count = 0
        mf.jump_to_start()

        for mevent in mf:
            if mevent.header.is_midas_internal_event():
                continue

            keys = mevent.banks.keys()
            
            # Scorriamo i banchi delle camere presente nell'evento MIDAS
            for key in keys:
                if key.startswith("CAM"):
                    # Determiniamo l'indice della camera (es. 'CAM0' -> 0, 'CAM1' -> 1)
                    try:
                        cam_idx = int(key.replace("CAM", ""))
                    except ValueError:
                        continue

                    if cam_idx >= num_cameras:
                        continue

                    # Estrazione della matrice 2D dell'immagine (height_single_cam, width)
                    img_arr, _, _ = cy.daq_cam2array(mevent.banks[key])

                    # Filtro di taglio superiore (es. cimax dal codice di ricostruzione)
                    if cimax is not None:
                        img_arr = np.where(img_arr < cimax, img_arr, 0)

                    # Calcolo offset Y per posizionare la camera nell'immagine complessiva
                    y_start = cam_idx * height_single_cam
                    y_end = y_start + height_single_cam

                    # Accumulo vettoriale velocissimo tramite NumPy
                    charge_sum[y_start:y_end, :] += img_arr
                    occupancy_sum[y_start:y_end, :] += (img_arr > 0).astype(np.float64)

            event_count += 1
            if event_count % 50 == 0:
                print(f"  > Processati {event_count} eventi MIDAS...")

        print(f"Run {run_num} completato ({event_count} eventi elaborati).")

    # Calcolo della mappa della MEDIA (Profile 2D = Charge / Occupancy)
    with np.errstate(divide='ignore', invalid='ignore'):
        profile_mean = np.where(occupancy_sum > 0, charge_sum / occupancy_sum, np.nan)
        charge_masked = np.where(occupancy_sum > 0, charge_sum, np.nan)
        occ_masked = np.where(occupancy_sum > 0, occupancy_sum, np.nan)

    # ---------------------------------------------------------
    # PLOTTING DEI PDF (Palette Rainbow + Centratura Perfetta)
    # ---------------------------------------------------------
    os.makedirs(output_pdf_dir, exist_ok=True)

    maps_to_plot = {
        "occupancy_2d": {
            "data": occ_masked,
            "title": "Pixel Occupancy Map (MIDAS Raw)",
            "zlabel": "Hits / Pixel",
            "log": False,
        },
        "integrated_charge_2d": {
            "data": charge_masked,
            "title": "Integrated Charge Map (MIDAS Raw)",
            "zlabel": "Total ADC",
            "log": False,
        },
        "profile_mean_2d": {
            "data": profile_mean,
            "title": "Mean Amplitude Profile 2D (MIDAS Raw)",
            "zlabel": "Mean ADC / Pixel",
            "log": False,
        },
    }

    x_edges = np.arange(0, width + 1)
    y_edges = np.arange(0, total_height + 1)

    for name, cfg in maps_to_plot.items():
        # Figsize proporzionata alla geometria verticale (3 camere impilate)
        fig, ax = plt.subplots(figsize=(7, 11))

        norm = LogNorm() if cfg["log"] else None

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            cfg["data"],
            shading="flat",
            cmap="rainbow",  # Palette richiesta
            norm=norm
        )

        cbar = fig.colorbar(mesh, ax=ax, pad=0.03, fraction=0.046)
        cbar.set_label(cfg["zlabel"], fontsize=12)

        ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("X [pixel]", fontsize=12)
        ax.set_ylabel("Y [pixel]", fontsize=12)

        # Mantiene le proporzioni geometriche reali dei pixel
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")

        output_path = os.path.join(output_pdf_dir, f"{name}.pdf")

        # bbox_inches='tight' rimuove completamente lo spazio bianco vuoto a sinistra
        plt.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        print(f"PDF Salvato e centrato correttamente: {output_path}")

if __name__ == "__main__":
    # Esempio di utilizzo:
    create_2dmaps_from_midas(
        run_start=124080,
        run_end=124080,
        output_pdf_dir="./plots_midas_pdf",
        tag="LNGS"
    )
