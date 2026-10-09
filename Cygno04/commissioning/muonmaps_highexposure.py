import os
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# DEFINIZIONE DATI DI INPUT
# key = (run-start, run-end, pedrun) : output_base_name
# -----------------------------------------------------------------------------
DATA = {
    (124650, 124650, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_000",
    (124649, 124649, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_040",
    (124648, 124648, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_140",
    (124647, 124647, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_240",
    (124646, 124646, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_340",
    (124645, 124645, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_440",
    (124644, 124644, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_540",
    (124643, 124643, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_640",
    (124642, 124642, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_740",
    (124641, 124641, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_840",
    (124640, 124640, 124639): "cygno04_sideA_exp7s_nostdcut_VGEM_460_VD_940",
}


def process_single_task(item):
    """
    Funzione worker eseguita in parallelo da ciascun core CPU.
    Esegue in sequenza i 3 comandi per la singola entry di DATA.
    """
    (rstart, rstop, pedrun), v = item
    prefix = f"[Run {rstart}-{rstop}]"
    
    print(f"\n{prefix} ===> INIZIO ELABORAZIONE per {v} (Pedrun: {pedrun})")

    # Step 1: Spike Veto
    cmd_spikeveto = (
        f"python analyze_midas_integral_vs_event.py --cutoff 50 "
        f"--run-start {rstart} --run-end {rstop} --pedrun {pedrun} "
        f"--xmin 200 --xmax 3800 --ymin 200 --ymax 6800"
    )
    print(f"{prefix} [Step 1/3] {cmd_spikeveto}")
    exit_code = os.system(cmd_spikeveto)
    if exit_code != 0:
        print(f"{prefix} [ERRORE] Fallita analisi spike per {v}")
        return False, v

    # Step 2: Make 2D Maps
    cmd_map = (
        f"python make_2dmaps_frommidas.py --nsigma -100.0 "
        f"--run-start {rstart} --run-end {rstop} --pedrun {pedrun} "
        f"--output {v}.root"
    )
    print(f"{prefix} [Step 2/3] {cmd_map}")
    exit_code = os.system(cmd_map)
    if exit_code != 0:
        print(f"{prefix} [ERRORE] Fallita creazione mappe per {v}")
        return False, v

    # Step 3: Plotting
    zmax = 55 if "VGEM_460" in v else (40 if "VGEM_450" in v else 25)
    cmd_plot = (
        f"python plot_2dmaps.py --input {v}.root "
        f"--output-dir cernplots --output-name {v} --zmax-prof {zmax}"
    )
    print(f"{prefix} [Step 3/3] {cmd_plot}")
    exit_code = os.system(cmd_plot)
    if exit_code != 0:
        print(f"{prefix} [ERRORE] Fallita generazione plot per {v}")
        return False, v

    print(f"{prefix} ===> COMPLETATO CON SUCCESSO: {v}")
    return True, v


def run_pipeline_parallel(n_jobs=1, dry_run=False):
    items = list(DATA.items())
    
    # Gestione del numero di core CPU
    max_cores = multiprocessing.cpu_count()
    if n_jobs <= 0 or n_jobs > max_cores:
        actual_workers = min(max_cores, len(items))
    else:
        actual_workers = min(n_jobs, len(items))

    print(f"========================================================")
    print(f"LANCIO PIPELINE SU {len(items)} TASK CON {actual_workers} CORE PARALLELI (disponibili: {max_cores})")
    print(f"========================================================")

    if dry_run:
        print("\n[MODALITÀ DRY-RUN] Nessun comando verrà eseguito.")
        for (rstart, rstop, pedrun), v in items:
            print(f"  > Task: {v} | Run-range: [{rstart}-{rstop}] | Pedestal: {pedrun}")
        return

    # Esecuzione parallela con ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(process_single_task, item): item for item in items}
        
        for future in as_completed(futures):
            success, name = future.result()
            status = "OK" if success else "FALLITO"
            print(f"--> [STATO TASK] {name}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue in parallelo la pipeline completa di analisi mappe MuonMap.")
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="Numero di core/processi paralleli (es. 4, oppure -1 per usarle tutte)")
    parser.add_argument("--dry-run", action="store_true", help="Stampa l'elenco dei task senza eseguirli")

    args = parser.parse_args()

    run_pipeline_parallel(n_jobs=args.jobs, dry_run=args.dry_run)

