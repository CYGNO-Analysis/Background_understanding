import os
import argparse
import numpy as np
import ROOT
import cygno as cy


def swift_download_midas_file(run, tmpdir, tag="LNGS"):
    print(f"--> Download/apertura file MIDAS per il run: {int(run):05d}")
    mfile = cy.open_mid(int(run), path=tmpdir, cloud=True, tag=tag, verbose=False)
    return mfile


def get_or_create_pedestal(
    pedrun,
    tag="LNGS",
    tmpdir="/tmp/",
    ped_dir="./pedestals",
    width=4096,
    height_single_cam=2304,
    num_cameras=3
):
    """
    Calcola (o carica dalla cache) le mappe 2D di Media e RMS/STD per il run di piedistallo.
    """
    os.makedirs(ped_dir, exist_ok=True)
    ped_cache_file = os.path.join(ped_dir, f"pedestal_run{pedrun:05d}.npz")

    # Se la mappa del piedistallo esiste già su disco, la carichiamo direttamente
    if os.path.exists(ped_cache_file):
        print(f"\n[PEDESTAL] Caricamento piedistallo da cache: {ped_cache_file}")
        data = np.load(ped_cache_file)
        return data["ped_mean"], data["ped_std"]

    print(f"\n[PEDESTAL] Calcolo piedistallo dal Run {pedrun:05d}...")
    total_height = height_single_cam * num_cameras

    # Accumulatori ad alta precisione per algoritmo Welford / Somma dei quadrati
    sum_img = np.zeros((total_height, width), dtype=np.float64)
    sum_sq_img = np.zeros((total_height, width), dtype=np.float64)
    counts = np.zeros((total_height, width), dtype=np.int64)

    mf = swift_download_midas_file(pedrun, tmpdir, tag)
    mf.jump_to_start()

    event_count = 0
    for mevent in mf:
        if mevent.header.is_midas_internal_event():
            continue

        for key in mevent.banks.keys():
            if key.startswith("CAM"):
                try:
                    cam_idx = int(key.replace("CAM", ""))
                except ValueError:
                    continue

                if cam_idx >= num_cameras:
                    continue

                img_arr, _, _ = cy.daq_cam2array(mevent.banks[key])
                img_arr = img_arr.astype(np.float64)

                y_start = cam_idx * height_single_cam
                y_end = y_start + height_single_cam

                sum_img[y_start:y_end, :] += img_arr
                sum_sq_img[y_start:y_end, :] += img_arr**2
                counts[y_start:y_end, :] += 1

        event_count += 1
        if event_count % 50 == 0:
            print(f"  > [Piedistallo] Processati {event_count} eventi...")

    # Calcolo Media e Std Deviazione per pixel
    np.maximum(counts, 1, out=counts) # Evita divisioni per zero
    ped_mean = sum_img / counts
    variance = (sum_sq_img / counts) - (ped_mean**2)
    ped_std = np.sqrt(np.maximum(variance, 0.0))

    # Salvataggio su file .npz per riutilizzi futuri ultra-veloci
    np.savez_compressed(ped_cache_file, ped_mean=ped_mean, ped_std=ped_std)
    print(f"[PEDESTAL] Mappe salvate in: {ped_cache_file}\n")

    return ped_mean, ped_std


def create_2dmaps_from_midas(
    run_start,
    run_end,
    pedrun,
    output_root_file="./midas_2dmaps.root",
    tag="LNGS",
    tmpdir="/tmp/",
    width=4096,
    height_single_cam=2304,
    num_cameras=3,
    nsigma=3.0,           # Soglia zero-suppression (default > 3*sigma)
    cimax=65535          # Soglia saturazione/artefatti
):
    total_height = height_single_cam * num_cameras

    # 1. Ottenimento mappe Piedistallo (Media e STD)
    ped_mean, ped_std = get_or_create_pedestal(
        pedrun=pedrun,
        tag=tag,
        tmpdir=tmpdir,
        width=width,
        height_single_cam=height_single_cam,
        num_cameras=num_cameras
    )

    # Accumulatori 2D (y, x) per l'intero detector
    charge_sum = np.zeros((total_height, width), dtype=np.float64)
    occupancy_sum = np.zeros((total_height, width), dtype=np.float64)

    # 2. Elaborazione dei Run di Segnale
    for run_num in range(run_start, run_end + 1):
        print(f"\n---> Elaborazione Run Segnale: {run_num:05d}")
        try:
            mf = swift_download_midas_file(run_num, tmpdir, tag)
        except Exception as e:
            print(f"ERRORE nel recupero del file MIDAS per il run {run_num}: {e}")
            continue

        event_count = 0
        mf.jump_to_start()

        for mevent in mf:
            if mevent.header.is_midas_internal_event():
                continue

            for key in mevent.banks.keys():
                if key.startswith("CAM"):
                    try:
                        cam_idx = int(key.replace("CAM", ""))
                    except ValueError:
                        continue

                    if cam_idx >= num_cameras:
                        continue

                    img_arr, _, _ = cy.daq_cam2array(mevent.banks[key])

                    y_start = cam_idx * height_single_cam
                    y_end = y_start + height_single_cam

                    # Sottrazione del piedistallo corrispondente a questa camera
                    img_sub = img_arr.astype(np.float64) - ped_mean[y_start:y_end, :]
                    ped_std_cam = ped_std[y_start:y_end, :]

                    # Zero suppression: considera solo i pixel con (pix - ped_mean) > nsigma * ped_std
                    # ed escludi eventuali artefatti/saturazioni oltre cimax
                    valid_mask = (img_sub > nsigma * ped_std_cam) & (img_arr < cimax)

                    # Azzeriamo i pixel sotto-soglia
                    img_clean = np.where(valid_mask, img_sub, 0.0)

                    # Accumulo vettoriale
                    charge_sum[y_start:y_end, :] += img_clean
                    occupancy_sum[y_start:y_end, :] += valid_mask.astype(np.float64)

            event_count += 1
            if event_count % 50 == 0:
                print(f"  > Processati {event_count} eventi MIDAS...")

        print(f"Run {run_num} completato ({event_count} eventi elaborati).")

    # 3. Salvataggio Mappe in File ROOT
    print("\nSalvataggio del file ROOT di output...")
    os.makedirs(os.path.dirname(os.path.abspath(output_root_file)), exist_ok=True)
    tfout = ROOT.TFile(output_root_file, "recreate")

    int2d  = ROOT.TH2D("int2d_images", "Integrated Charge Map (Pedestal Subtracted)", 256, 0, width, 144 * num_cameras, 0, total_height)
    occ2d  = ROOT.TH2D("occ2d_images", "Pixel Occupancy Map (Zero Suppressed)", 256, 0, width, 144 * num_cameras, 0, total_height)

    # Estrazione coordinate dei pixel con occupancy > 0
    y_indices, x_indices = np.where(occupancy_sum > 0)
    
    x_coords = x_indices.astype(np.float64) + 0.5
    y_coords = y_indices.astype(np.float64) + 0.5
    
    weights_occ = occupancy_sum[y_indices, x_indices].astype(np.float64)
    weights_int = charge_sum[y_indices, x_indices].astype(np.float64)

    n_points = len(x_coords)
    if n_points > 0:
        occ2d.FillN(n_points, x_coords, y_coords, weights_occ)
        int2d.FillN(n_points, x_coords, y_coords, weights_int)

    # Media = Carica Totale / Occupancy
    prof2d = int2d.Clone("prof2d_images")
    prof2d.Divide(occ2d)

    tfout.cd()
    occ2d.Write()
    int2d.Write()
    prof2d.Write()
    tfout.Close()

    print(f"FILE ROOT Salvato con successo: {output_root_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mappe 2D MIDAS con sottrazione piedistallo e zero suppression")
    parser.add_argument("--run-start", type=int, required=True, help="Run di segnale iniziale")
    parser.add_argument("--run-end", type=int, required=True, help="Run di segnale finale")
    parser.add_argument("--pedrun", type=int, required=True, help="Run MIDAS di piedistallo da usare")
    parser.add_argument("--output", type=str, default="./midas_2dmaps_pedsub.root", help="Path del file ROOT di output")
    parser.add_argument("--nsigma", type=float, default=3.0, help="Soglia n*sigma per la zero suppression (default: 3.0)")
    parser.add_argument("--tag", type=str, default="LNGS", help="Tag CYGNO per download cloud")
    parser.add_argument("--tmpdir", type=str, default="/tmp/", help="Directory temporanea")

    args = parser.parse_args()

    create_2dmaps_from_midas(
        run_start=args.run_start,
        run_end=args.run_end,
        pedrun=args.pedrun,
        output_root_file=args.output,
        nsigma=args.nsigma,
        tag=args.tag,
        tmpdir=args.tmpdir
    )
