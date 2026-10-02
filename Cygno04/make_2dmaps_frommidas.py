import os
import argparse
import numpy as np
import ROOT
import cygno as cy


def swift_download_midas_file(run, tmpdir, tag="LNGS"):
    print("download or open midas file for run ", int(run))
    mfile = cy.open_mid(int(run), path=tmpdir, cloud=True, tag=tag, verbose=True)
    return mfile


def create_2dmaps_from_midas(
    run_start,
    run_end,
    output_root_file="./midas_2dmaps.root",
    tag="LNGS",
    tmpdir="/tmp/",
    width=4096,
    height_single_cam=2304,
    num_cameras=3,
    cimax=65535,         # Threshold massimo per escludere saturazione/artefatti
    pedestal_file=None   # Opzionale: array o percorso se si vuole sottrarre il piedistallo
):
    total_height = height_single_cam * num_cameras

    # Accumulatori 2D (y, x) per l'intero detector
    charge_sum = np.zeros((total_height, width), dtype=np.float64)
    occupancy_sum = np.zeros((total_height, width), dtype=np.float64)

    for run_num in range(run_start, run_end + 1):
        print(f"\n---> Scaricamento/Apertura MIDAS per Run: {run_num:05d}")
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

            keys = mevent.banks.keys()
            
            # Scorriamo i banchi delle camere presenti nell'evento MIDAS
            for key in keys:
                if key.startswith("CAM"):
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

    # ---------------------------------------------------------
    # SALVATAGGIO MAPPE IN FILE ROOT (Compatibile con plotter)
    # ---------------------------------------------------------
    print("\nCreazione file ROOT di output e riempimento istogrammi 2D...")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_root_file)), exist_ok=True)
    tfout = ROOT.TFile(output_root_file, "recreate")

    # Inizializziamo gli stessi istogrammi usati dallo script dai cluster
    prof2d = ROOT.TH2D("prof2d", "Mean Amplitude Profile 2D", 256, 0, width, 144 * num_cameras, 0, total_height)
    int2d  = ROOT.TH2D("int2d", "Integrated Charge Map", 256, 0, width, 144 * num_cameras, 0, total_height)
    occ2d  = ROOT.TH2D("occ2d", "Pixel Occupancy Map", 256, 0, width, 144 * num_cameras, 0, total_height)

    # Estraiamo solo i punti dove l'occupancy è > 0 per salvataggio leggero e veloce
    y_indices, x_indices = np.where(occupancy_sum > 0)
    
    # Inizializziamo i dati centro-bin dei pixel (aggiungiamo +0.5 per posizionare al centro del pixel)
    x_coords = x_indices.astype(np.float64) + 0.5
    y_coords = y_indices.astype(np.float64) + 0.5
    
    weights_occ = occupancy_sum[y_indices, x_indices].astype(np.float64)
    weights_int = charge_sum[y_indices, x_indices].astype(np.float64)

    # FillN C++ ultra-veloce su tutto l'array
    n_points = len(x_coords)
    if n_points > 0:
        occ2d.FillN(n_points, x_coords, y_coords, weights_occ)
        int2d.FillN(n_points, x_coords, y_coords, weights_int)

    # Calcolo della media (Profile 2D = Integrated / Occupancy) direttamente in ROOT
    prof2d.Divide(int2d, occ2d)

    # Scrittura ed eliminazione overhead
    tfout.cd()
    occ2d.Write()
    int2d.Write()
    prof2d.Write()
    tfout.Close()

    print(f"FILE ROOT Salvato con successo: {output_root_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrae mappe 2D da file MIDAS e le salva in un ROOT File")
    parser.add_argument("--run-start", type=int, required=True, help="Numero di Run iniziale")
    parser.add_argument("--run-end", type=int, required=True, help="Numero di Run finale")
    parser.add_argument("--output", type=str, default="./midas_2dmaps.root", help="Path del file ROOT di output")
    parser.add_argument("--tag", type=str, default="LNGS", help="Tag CYGNO per lo scaricamento cloud")
    parser.add_argument("--tmpdir", type=str, default="/tmp/", help="Directory temporanea per file MIDAS")

    args = parser.parse_args()

    create_2dmaps_from_midas(
        run_start=args.run_start,
        run_end=args.run_end,
        output_root_file=args.output,
        tag=args.tag,
        tmpdir=args.tmpdir
    )
