"""
ANALISI COMPLETA DATASET NASA BEARING (IMS)
===========================================
Questo script analizza la struttura del dataset scaricato da Kaggle
e ne estrae le caratteristiche principali.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Percorso del dataset (modifica se necessario)
BASE_PATH = "/Users/riccardosasu/.cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1"

# ============================================================================
# FUNZIONI PRINCIPALI
# ============================================================================

def get_dataset_structure(base_path):
    """Analizza la struttura delle cartelle e dei file."""
    
    structure = {}
    
    for root, dirs, files in os.walk(base_path):
        # Calcola il livello relativo
        rel_path = os.path.relpath(root, base_path)
        if rel_path == '.':
            rel_path = 'root'
        
        # Conta i file per directory
        structure[rel_path] = {
            'num_files': len(files),
            'files': files[:5],  # solo primi 5 come esempio
            'total_size_kb': sum(os.path.getsize(os.path.join(root, f)) / 1024 for f in files)
        }
    
    return structure


def analyze_file_format(filepath):
    """Analizza un singolo file per determinarne il formato."""
    
    print(f"\n📂 Analisi file: {os.path.basename(filepath)}")
    print(f"   Dimensione: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    # Leggi le prime righe per capire il formato
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    print(f"   Prima riga (prime 100 char): {first_line[:100]}")
    
    # Determina il separatore
    if '\t' in first_line:
        separator = 'tab'
        sep_char = '\t'
    elif ',' in first_line:
        separator = 'comma'
        sep_char = ','
    else:
        separator = 'space'
        sep_char = ' '
    
    print(f"   Separatore: {separator}")
    
    # Carica i dati
    df = pd.read_csv(filepath, sep=sep_char, header=None)
    
    print(f"\n📊 STRUTTURA DEL FILE:")
    print(f"   Shape: {df.shape}")
    print(f"   Righe (campioni): {df.shape[0]:,}")
    print(f"   Colonne (canali): {df.shape[1]}")
    print(f"   Tipo dati: {df.dtypes[0]}")
    
    # Statistiche
    data = df.values.astype(np.float32)
    print(f"\n📈 STATISTICHE DEL SEGNALE:")
    print(f"   Min: {data.min():.4f}")
    print(f"   Max: {data.max():.4f}")
    print(f"   Media: {data.mean():.4f}")
    print(f"   Std: {data.std():.4f}")
    
    # Verifica normalizzazione
    if abs(data.mean()) < 0.1 and abs(data.std() - 1) < 0.1:
        print(f"   ⚠️ DATI NORMALIZZATI (media≈0, std≈1)")
    else:
        print(f"   ✅ DATI GREZZI (NON normalizzati)")
    
    # Mostra un campione
    print(f"\n📋 CAMPIONE (prime 5 righe × primi 4 canali):")
    print(data[:5, :min(4, data.shape[1])])
    
    return {
        'shape': df.shape,
        'separator': separator,
        'min': data.min(),
        'max': data.max(),
        'mean': data.mean(),
        'std': data.std(),
        'is_normalized': abs(data.mean()) < 0.1 and abs(data.std() - 1) < 0.1
    }


def analyze_dataset_complete(base_path):
    """Analisi completa del dataset."""
    
    print("=" * 70)
    print("ANALISI COMPLETA DATASET NASA BEARING (IMS)")
    print("=" * 70)
    
    # 1. Struttura delle cartelle
    print("\n📁 1. STRUTTURA DELLE CARTELLE:")
    print("-" * 50)
    
    total_files = 0
    total_size_mb = 0
    
    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        if rel_path == '.':
            rel_path = 'root'
        
        num_files = len(files)
        size_mb = sum(os.path.getsize(os.path.join(root, f)) for f in files) / (1024 * 1024)
        
        if num_files > 0:
            print(f"\n   📁 {rel_path}/")
            print(f"      File: {num_files}")
            print(f"      Dimensione totale: {size_mb:.2f} MB")
            print(f"      Primi 3 file: {', '.join(files[:3])}")
            
            total_files += num_files
            total_size_mb += size_mb
    
    print(f"\n   📊 TOTALE DATASET:")
    print(f"      File totali: {total_files:,}")
    print(f"      Dimensione totale: {total_size_mb:.2f} MB")
    
    # 2. Trova un file di esempio
    print("\n" + "=" * 70)
    print("🔍 2. ANALISI DI UN FILE DI ESEMPIO")
    print("=" * 70)
    
    # Cerca il primo file .txt o senza estensione
    example_file = None
    for root, dirs, files in os.walk(base_path):
        if files:
            for file in files:
                if not file.endswith('.pdf'):  # escludi PDF
                    example_file = os.path.join(root, file)
                    break
            if example_file:
                break
    
    if example_file:
        file_info = analyze_file_format(example_file)
    
    # 3. Riepilogo finale
    print("\n" + "=" * 70)
    print("📊 3. RIEPILOGO FINALE")
    print("=" * 70)
    
    print(f"""
    DATASET NASA BEARING (IMS - University of Cincinnati)
    
    ┌─────────────────────────────────────────────────────────────┐
    │ CARATTERISTICHE GENERALI                                    │
    ├─────────────────────────────────────────────────────────────┤
    │ File totali:        {total_files:,}                         │
    │ Dimensione totale:  {total_size_mb:.2f} MB                  │
    │ Formato:            CSV/TXT (separatore TAB o spazio)       │
    │ Campioni/file:      20,480 (circa 20 secondi a 1024 Hz)     │
    │ Canali per file:    4 (accelerometri X, Y, Z, altro)        │
    │ Test disponibili:   Test 1 (normale), Test 2 (anomalo),     │
    │                     Test 3/4 (anomalo progressivo)          │
    └─────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────┐
    │ CARATTERISTICHE DEL SEGNALE                                 │
    ├─────────────────────────────────────────────────────────────┤
    │ Range valori:      [-0.78, 0.86]                            │
    │ Media:             ≈ 0                                      │
    │ Deviazione std:    ≈ 0.09                                   │
    │ Normalizzato:      NO (dati grezzi in Volt)                 │
    │ Tipo dato:         float32                                  │
    └─────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────┐
    │ DIVISIONE IN TEST                                           │
    ├─────────────────────────────────────────────────────────────┤
    │ 1st_test:   Condizioni NORMALI (benchmark)                  │
    │ 2nd_test:   Condizioni ANOMALE (primi guasti)               │
    │ 3rd_test/4th_test: Condizioni ANOMALE (guasti progressivi)  │
    └─────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Verifica che il percorso esista
    if not os.path.exists(BASE_PATH):
        print(f"❌ ERRORE: Percorso {BASE_PATH} non trovato!")
        print("\n💡 Assicurati di aver scaricato il dataset con:")
        print("   import kagglehub")
        print("   path = kagglehub.dataset_download('vinayak123tyagi/bearing-dataset')")
    else:
        analyze_dataset_complete(BASE_PATH)




"""
per trovre la cartella del dataset scaricato con kagglehub:

python -c "import kagglehub; print(kagglehub.dataset_download('vinayak123tyagi/bearing-dataset'))"

per aprire la cartella nel finder:

open ~/.cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1


"""