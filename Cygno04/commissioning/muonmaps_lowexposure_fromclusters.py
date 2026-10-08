import pandas as pd
import argparse
import subprocess
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_consecutive_run_ranges(df):
    """
    Raggruppa run consecutivi che hanno gli stessi valori di GEM1_V e DRIFT_V.
    """
    if df.empty:
        return []

    # Ordina per numero di run per garantire la sequenzialità
    df = df.sort_values("run_number").reset_index(drop=True)

    ranges = []
    current_start = df.loc[0, "run_number"]
    current_end = df.loc[0, "run_number"]
    current_vgem = df.loc[0, "GEM1_V"]
    current_vdrift = df.loc[0, "DRIFT_V"]

    for i in range(1, len(df)):
        row = df.loc[i]
        run_num = row["run_number"]
        vgem = row["GEM1_V"]
        vdrift = row["DRIFT_V"]

        if (run_num == current_end + 1) and (vgem == current_vgem) and (vdrift == current_vdrift):
            current_end = run_num
        else:
            ranges.append({
                "run_min": int(current_start),
                "run_max": int(current_end),
                "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
                "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
            })
            current_start = run_num
            current_end = run_num
            current_vgem = vgem
            current_vdrift = vdrift

    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
    })

    return ranges


def execute_command(task):
    """
    Funzione worker eseguita in parallelo da ciascun processo CPU.
    """
    run_min, run_max, vgem, vdrift, cmd = task
    cmd_str = " ".join(cmd)
    print(f"\n[AVVIO CORE] Range {run_min}-{run_max} (VGEM={vgem}V, VD={vdrift}V)\nComando: {cmd_str}")

    try:
        subprocess.run(cmd, check=True)
        print(f"[COMPLETATO] Range {run_min}-{run_max}")
        return True, run_min, run_max
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] Fallito range {run_min}-{run_max}: {e}")
        return False, run_min, run_max


def run_map_processing(csv_file, execute=False, n_jobs=1):
    print(f"--> Lettura file CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    df["run_description"] = df["run_description"].astype(str).str.strip()
    df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()

    mask = (df["run_description"] == "MuonMap") & (df["file_s3_tag"] == "LNGS")
    df_filtered = df[mask].copy()

    print(f"--> Trovati {len(df_filtered)} run con run_description='MuonMap' e file_s3_tag='LNGS'.")

    if df_filtered.empty:
        print("Nessun run trovato con i criteri specificati. Uscita.")
        return

    df_filtered["run_number"] = pd.to_numeric(df_filtered["run_number"], errors="coerce")
    df_filtered["GEM1_V"] = pd.to_numeric(df_filtered["GEM1_V"], errors="coerce")
    df_filtered["DRIFT_V"] = pd.to_numeric(df_filtered["DRIFT_V"], errors="coerce")

    run_ranges = find_consecutive_run_ranges(df_filtered)

    print(f"\n========================================================")
    print(f"IDENTIFICATI {len(run_ranges)} RUN RANGES CONSECUTIVI")
    print(f"========================================================")

    tasks = []
    for r in run_ranges:
        run_min = r["run_min"]
        run_max = r["run_max"]
        vgem = r["vgem"]
        vdrift = r["vdrift"]

        output_filename = f"cygno04_sideA_muons_VGEM_{vgem}_VD_{vdrift}.root"
        
        cmd = [
            "python", "make_2dmaps_fromclusters_s3.py",
            "--run-start", str(run_min),
            "--run-end", str(run_max),
            "--output", output_filename
        ]

        tasks.append((run_min, run_max, vgem, vdrift, cmd))
        print(f"  > Range {run_min}-{run_max} | VGEM={vgem}V, VD={vdrift}V -> {output_filename}")

    if not execute:
        print("\n(Modalità DRY-RUN: Usa --execute per lanciare i comandi)")
        return

    # Determinazione dei thread/processi da usare
    max_cores = multiprocessing.cpu_count()
    if n_jobs <= 0 or n_jobs > max_cores:
        n_jobs = max_cores

    actual_workers = min(n_jobs, len(tasks))
    print(f"\n---> ESECUZIONE PARALLELA SU {actual_workers} CORE CPU (disponibili: {max_cores})...\n")

    # Esecuzione in parallelo con ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = [executor.submit(execute_command, task) for task in tasks]
        for future in as_completed(futures):
            success, rmin, rmax = future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa file CSV e lancia in parallelo sui core CPU la creazione delle mappe.")
    parser.add_argument("--csv", type=str, required=True, help="Path del file CSV di input")
    parser.add_argument("--execute", action="store_true", help="Esegue effettivamente i comandi")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Numero di job/processi paralleli (es. 4, o -1 per usare tutti i core dell'M4)")

    args = parser.parse_args()

    run_map_processing(csv_file=args.csv, execute=args.execute, n_jobs=args.jobs)
