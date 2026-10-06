import os
import argparse
import numpy as np
import ROOT
import cygno as cy
import torch

def get_device():
    """Rileva e restituisce il device MPS (Metal GPU) su Apple Silicon M4."""
    if torch.backends.mps.is_available():
        print("[GPU] Trovato acceleratore Apple Silicon GPU (MPS Metal)")
        return torch.device("mps")
    print("[CPU] GPU MPS non disponibile, uso CPU standard")
    return torch.device("cpu")


def swift_download_midas_file(run, tmpdir, tag="LNGS"):
    print(f"--> Download/apertura file MIDAS per il run: {int(run):05d}")
    return cy.open_mid(int(run), path=tmpdir, cloud=True, tag=tag, verbose=False)


def get_or_create_pedestal(
    pedrun,
    device,
    tag="LNGS",
    tmpdir="/tmp/",
    ped_dir="./pedestals",
    width=4096,
    height_single_cam=2304,
    num_cameras=3
):
    os.makedirs(ped_dir, exist_ok=True)
    ped_cache_file = os.path.join(ped_dir, f"pedestal_run{pedrun:05d}.npz")

    if os.path.exists(ped_cache_file):
        print(f"\n[PEDESTAL] Caricamento piedistallo da cache: {ped_cache_file}")
        data = np.load(ped_cache_file)
        ped_mean_t = torch.from_numpy(data["ped_mean"]).to(device=device, dtype=torch.float32)
        ped_std_t = torch.from_numpy(data["ped_std"]).to(device=device, dtype=torch.float32)
        return ped_mean_t, ped_std_t

    print(f"\n[PEDESTAL] Calcolo piedistallo dal Run {pedrun:05d} (GPU M4)...")
    total_height = height_single_cam * num_cameras

    # Accumulatori PyTorch su GPU Metal
    sum_img = torch.zeros((total_height, width), dtype=torch.float64, device=device)
    sum_sq_img = torch.zeros((total_height, width), dtype=torch.float64, device=device)
    counts = torch.zeros((total_height, width), dtype=torch.int64, device=device)

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
                img_t = torch.from_numpy(img_arr).to(device=device, dtype=torch.float64)

                y_start = cam_idx * height_single_cam
                y_end = y_start + height_single_cam

                sum_img[y_start:y_end, :] += img_t
                sum_sq_img[y_start:y_end, :] += img_t**2
                counts[y_start:y_end, :] += 1

        event_count += 1
        if event_count % 10 == 0:
            print(f"  > [Piedistallo] Processati {event_count} eventi...")

    counts = torch.clamp(counts, min=1)
    ped_mean = sum_img / counts
    variance = (sum_sq_img / counts) - (ped_mean**2)
    ped_std = torch.sqrt(torch.clamp(variance, min=0.0))

    # Convertiamo in CPU / NumPy per salvataggio npz
    ped_mean_np = ped_mean.cpu().numpy().astype(np.float32)
    ped_std_np = ped_std.cpu().numpy().astype(np.float32)
    np.savez_compressed(ped_cache_file, ped_mean=ped_mean_np, ped_std=ped_std_np)

    return ped_mean.to(dtype=torch.float32), ped_std.to(dtype=torch.float32)


def create_2dmaps_from_midas(
    run_start,
    run_end,
    pedrun,
    output_root_file="./midas_2dmaps_pedsub.root",
    tag="LNGS",
    tmpdir="/tmp/",
    width=4096,
    height_single_cam=2304,
    num_cameras=3,
    nsigma=3.0,
    cimax=65535
):
    device = get_device()
    total_height = height_single_cam * num_cameras

    # 1. Caricamento Mappe Piedistallo direttamente in memoria GPU MPS
    ped_mean, ped_std = get_or_create_pedestal(
        pedrun=pedrun,
        device=device,
        tag=tag,
        tmpdir=tmpdir,
        width=width,
        height_single_cam=height_single_cam,
        num_cameras=num_cameras
    )

    # Accumulatori GPU
    charge_sum = torch.zeros((total_height, width), dtype=torch.float32, device=device)
    occupancy_sum = torch.zeros((total_height, width), dtype=torch.float32, device=device)

    # Pre-calcolo soglia nsigma * std in GPU
    nsigma_std = nsigma * ped_std

    # 2. Elaborazione dei Run
    for run_num in range(run_start, run_end + 1):
        print(f"\n---> Elaborazione Run Segnale: {run_num:05d}")
        try:
            mf = swift_download_midas_file(run_num, tmpdir, tag)
        except Exception as e:
            print(f"ERRORE file MIDAS {run_num}: {e}")
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

                    # Spostamento immediato della matrice 2D nella memoria unificata GPU Apple
                    img_t = torch.from_numpy(img_arr).to(device=device, dtype=torch.float32)

                    ped_mean_cam = ped_mean[y_start:y_end, :]
                    nsigma_std_cam = nsigma_std[y_start:y_end, :]

                    # Operazioni vettoriali parallele sui core GPU
                    img_sub = img_t - ped_mean_cam
                    valid_mask = (img_sub > nsigma_std_cam) & (img_t < cimax)

                    img_clean = torch.where(valid_mask, img_sub, 0.0)

                    # Accumulo su memoria Metal
                    charge_sum[y_start:y_end, :] += img_clean
                    occupancy_sum[y_start:y_end, :] += valid_mask.to(torch.float32)

            event_count += 1
            if event_count % 10 == 0:
                print(f"  > Processati {event_count} eventi MIDAS su GPU M4...")

    # Trasferimento finale dei risultati da GPU MPS a CPU per salvare il ROOT file
    charge_sum_np = charge_sum.cpu().numpy()
    occupancy_sum_np = occupancy_sum.cpu().numpy()

    # 3. Salvataggio ROOT
    print("\nSalvataggio del file ROOT di output...")
    os.makedirs(os.path.dirname(os.path.abspath(output_root_file)), exist_ok=True)
    tfout = ROOT.TFile(output_root_file, "recreate")

    int2d = ROOT.TH2D("int2d_images", "Integrated Charge Map", 256, 0, width, 144 * num_cameras, 0, total_height)
    occ2d = ROOT.TH2D("occ2d_images", "Pixel Occupancy Map", 256, 0, width, 144 * num_cameras, 0, total_height)

    y_indices, x_indices = np.where(occupancy_sum_np > 0)
    x_coords = x_indices.astype(np.float64) + 0.5
    y_coords = y_indices.astype(np.float64) + 0.5

    weights_occ = occupancy_sum_np[y_indices, x_indices].astype(np.float64)
    weights_int = charge_sum_np[y_indices, x_indices].astype(np.float64)

    n_points = len(x_coords)
    if n_points > 0:
        occ2d.FillN(n_points, x_coords, y_coords, weights_occ)
        int2d.FillN(n_points, x_coords, y_coords, weights_int)

    prof2d = int2d.Clone("prof2d_images")
    prof2d.Divide(occ2d)

    tfout.cd()
    occ2d.Write()
    int2d.Write()
    prof2d.Write()
    tfout.Close()

    print(f"FILE ROOT Salvato con successo: {output_root_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mappe 2D MIDAS con accelerazione GPU M4 PyTorch/Metal")
    parser.add_argument("--run-start", type=int, required=True)
    parser.add_argument("--run-end", type=int, required=True)
    parser.add_argument("--pedrun", type=int, required=True)
    parser.add_argument("--output", type=str, default="./midas_2dmaps_pedsub.root")
    parser.add_argument("--nsigma", type=float, default=3.0)
    parser.add_argument("--tag", type=str, default="LNGS")
    parser.add_argument("--tmpdir", type=str, default="/tmp/")

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
