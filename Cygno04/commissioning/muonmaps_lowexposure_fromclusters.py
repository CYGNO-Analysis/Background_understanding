import pandas as pd
import argparse
import subprocess
import os

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

        # Se il run è consecutivo al precedente E ha le stesse tensioni, estendiamo il range
        if (run_num == current_end + 1) and (vgem == current_vgem) and (vdrift == current_vdrift):
            current_end = run_num
        else:
            # Salva il range precedente e inizia uno nuovo
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

    # Aggiungi l'ultimo gruppo
    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
    })

    return ranges


def run_map_processing(csv_file, execute=False):
    # 1. Lettura del CSV
    print(f"--> Lettura file CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    # Clean delle stringhe per evitare problemi con spazi o maiuscole/minuscole
    df["run_description"] = df["run_description"].astype(str).str.strip()
    df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()

    # 2. Filtraggio dei dati
    mask = (df["run_description"] == "MuonMap") & (df["file_s3_tag"] == "LNGS")
    df_filtered = df[mask].copy()

    print(f"--> Trovati {len(df_filtered)} run con run_description='MuonMap' e file_s3_tag='LNGS'.")

    if df_filtered.empty:
        print("Nessun run trovato con i criteri specificati. Uscita.")
        return

    # Converti colonne di interesse in numerico
    df_filtered["run_number"] = pd.to_numeric(df_filtered["run_number"], errors="coerce")
    df_filtered["GEM1_V"] = pd.to_numeric(df_filtered["GEM1_V"], errors="coerce")
    df_filtered["DRIFT_V"] = pd.to_numeric(df_filtered["DRIFT_V"], errors="coerce")

    # 3. Identificazione dei Run Ranges
    run_ranges = find_consecutive_run_ranges(df_filtered)

    print(f"\n========================================================")
    print(f"IDENTIFICATI {len(run_ranges)} RUN RANGES CONSECUTIVI:")
    print(f"========================================================")

    # 4. Generazione ed esecuzione dei comandi
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

        cmd_str = " ".join(cmd)
        print(f"\n[Range {run_min}-{run_max}] VGEM={vgem}V, Vdrift={vdrift}V")
        print(f"  Comando: {cmd_str}")

        if execute:
            print(f"  --> Esecuzione in corso...")
            try:
                subprocess.run(cmd, check=True)
                print("  --> Completato con successo!")
            except subprocess.CalledProcessError as e:
                print(f"  ERRORE durante l'esecuzione del comando: {e}")
        else:
            print("  (Modalità DRY-RUN: Usa --execute per lanciare effettivamente i comandi)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa file CSV per identificare range di run di MuonMap e genera mappe.")
    parser.add_argument("--csv", type=str, required=True, help="Path del file CSV di input (logbook)")
    parser.add_argument("--execute", action="store_true", help="Esegue effettivamente i comandi (default: solo dry-run)")

    args = parser.parse_args()

    run_map_processing(csv_file=args.csv, execute=args.execute)
