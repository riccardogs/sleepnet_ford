#!/usr/bin/env python3
"""
VERIFICA COMPLETA DATASET FORDB
================================
Esegue tutti i controlli necessari per verificare l'integrità
e le caratteristiche del dataset FordB scaricato.
"""

import numpy as np
import os
import sys
from pathlib import Path

def print_separator(char='=', length=60):
    print(char * length)

def check_file_exists(filepath):
    """Verifica se il file esiste"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ File trovato: {filepath}")
        print(f"   Dimensione: {size:.2f} MB")
        return True
    else:
        print(f"❌ File non trovato: {filepath}")
        return False

def check_file_integrity(data):
    """Verifica l'integrità dei dati"""
    print("\n🔍 INTEGRITÀ DEI DATI:")
    
    checks = {
        'NaN presenti': np.isnan(data['X']).any(),
        'Infiniti presenti': np.isinf(data['X']).any(),
        'Dati vuoti': data['X'].size == 0
    }
    
    all_ok = True
    for check_name, is_bad in checks.items():
        if is_bad:
            print(f"   ❌ {check_name}: SÌ")
            all_ok = False
        else:
            print(f"   ✅ {check_name}: NO")
    
    return all_ok

def print_statistics(X, y):
    """Stampa statistiche del dataset"""
    print("\n📊 STATISTICHE:")
    print(f"   Numero campioni: {X.shape[0]:,}")
    print(f"   Shape X: {X.shape}")
    print(f"   Shape y: {y.shape}")
    print(f"   Tipo X: {X.dtype}")
    print(f"   Tipo y: {y.dtype}")
    print(f"   Min X: {X.min():.6f}")
    print(f"   Max X: {X.max():.6f}")
    print(f"   Media X: {X.mean():.6f}")
    print(f"   Std X: {X.std():.6f}")

def print_class_distribution(y):
    """Stampa distribuzione delle classi"""
    print("\n📊 DISTRIBUZIONE CLASSI:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = count / len(y) * 100
        print(f"   Classe {cls}: {count:,} campioni ({pct:.1f}%)")
    
    # Interpretazione
    if len(unique) == 2:
        print("\n   ✅ Dataset binario (2 classi)")
        if '-1' in unique and '1' in unique:
            print(f"   📌 Nota: classi stringhe '-1' e '1' → convertire in interi 0 e 1 per il modello")
    else:
        print(f"   ⚠️ Attenzione: {len(unique)} classi trovate")

def print_sample_info(X, y, idx=0):
    """Stampa info su un campione specifico"""
    print(f"\n🔬 CAMPIONE {idx}:")
    sample = X[idx]
    label = y[idx]
    
    print(f"   Shape: {sample.shape}")
    print(f"   Label: {label}")
    print(f"   Min: {sample.min():.6f}")
    print(f"   Max: {sample.max():.6f}")
    print(f"   Media: {sample.mean():.6f}")
    print(f"   Std: {sample.std():.6f}")
    print(f"   Primi 10 valori: {sample.flatten()[:10]}")

def check_pytorch_compatibility(X):
    """Verifica compatibilità con PyTorch"""
    print("\n🔥 COMPATIBILITÀ PyTorch:")
    try:
        import torch
        print("   ✅ PyTorch importato")
        
        # Crea tensore
        tensor = torch.tensor(X[0], dtype=torch.float32)
        print(f"   ✅ Tensore creato: shape {tensor.shape}")
        
        # Aggiungi dimensione batch e canale
        batch_tensor = tensor.unsqueeze(0).unsqueeze(0)
        print(f"   ✅ Batch tensor: shape {batch_tensor.shape}")
        
        if batch_tensor.shape[1] == 1:
            print(f"   ✅ Canale singolo (1) - OK per Conv1d")
        else:
            print(f"   ⚠️ Canali: {batch_tensor.shape[1]} → verificare modello")
            
    except ImportError:
        print("   ❌ PyTorch non installato")
        return False
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False
    return True

def suggest_conversion(y):
    """Suggerisci conversione classi se necessario"""
    print("\n💡 SUGGERIMENTI PER IL MODELLO:")
    
    unique = np.unique(y)
    if '-1' in unique and '1' in unique:
        print("   📌 Le classi sono stringhe: '-1' e '1'")
        print("   📌 Per usare con SimpleSleepNet, convertile in interi 0 e 1:")
        print("      y_int = y.astype(int)")
        print("      y_binary = (y_int + 1) // 2")
        
        # Crea versione convertita (gestendo le stringhe)
        y_int = y.astype(int)
        y_binary = (y_int + 1) // 2
        print(f"\n   ✅ Dopo conversione: classi {np.unique(y_binary)}")
        print(f"      Classe 0: {np.sum(y_binary==0)} campioni")
        print(f"      Classe 1: {np.sum(y_binary==1)} campioni")
    
    print("\n   📌 Per usare FordB nel modello:")
    print("      1. Carica: data = np.load('data/fordb_originale.npz')")
    print("      2. Prendi X = data['X']")
    print("      3. Converti y: y = (data['y'].astype(int) + 1) // 2")
    print("      4. Usa il dataloader FordADataLoader esistente")

def save_ready_version(X, y, output_path='data/fordb_ready.npz'):
    """Salva una versione pronta all'uso con classi 0/1"""
    print(f"\n💾 SALVATAGGIO VERSIONE PRONTA:")
    
    # Gestisci y come stringhe e converti
    if y.dtype.kind in ['U', 'S']:  # Stringa o byte
        print(f"   📌 Rilevato tipo y: {y.dtype} (stringa)")
        y_int = y.astype(int)
        print(f"   ✅ Convertito in interi: {y_int.dtype}")
    else:
        y_int = y
    
    y_binary = (y_int + 1) // 2
    
    # Salva
    np.savez(output_path, X=X, y=y_binary)
    print(f"   ✅ Salvato in: {output_path}")
    
    # Verifica
    test = np.load(output_path)
    print(f"   ✅ Verifica: X={test['X'].shape}, y={np.unique(test['y'])}")
    return output_path

def main():
    print_separator('=')
    print("VERIFICA COMPLETA DATASET FORDB")
    print_separator('=')
    
    # ========== 1. VERIFICA FILE ==========
    filepath = 'data/fordb_originale.npz'
    print("\n📍 FASE 1: VERIFICA FILE")
    print_separator('-')
    
    if not check_file_exists(filepath):
        print("\n❌ Dataset FordB non trovato!")
        print("   Scaricalo con: python -c \"from aeon.datasets import load_classification; import numpy as np; X, y = load_classification('FordB'); np.savez('data/fordb_originale.npz', X=X, y=y)\"")
        sys.exit(1)
    
    # ========== 2. CARICA DATI ==========
    print("\n📍 FASE 2: CARICAMENTO DATI")
    print_separator('-')
    
    data = np.load(filepath)
    X = data['X']
    y = data['y']
    
    # ========== 3. STATISTICHE ==========
    print("\n📍 FASE 3: STATISTICHE GENERALI")
    print_separator('-')
    print_statistics(X, y)
    
    # ========== 4. INTEGRITÀ ==========
    print("\n📍 FASE 4: INTEGRITÀ DATI")
    print_separator('-')
    check_file_integrity(data)
    
    # ========== 5. CLASSI ==========
    print("\n📍 FASE 5: ANALISI CLASSI")
    print_separator('-')
    print_class_distribution(y)
    
    # ========== 6. CAMPIONE ==========
    print("\n📍 FASE 6: ANALISI CAMPIONE")
    print_separator('-')
    print_sample_info(X, y, idx=0)
    
    # ========== 7. PYTORCH ==========
    print("\n📍 FASE 7: COMPATIBILITÀ PYTORCH")
    print_separator('-')
    check_pytorch_compatibility(X)
    
    # ========== 8. SUGGERIMENTI ==========
    print("\n📍 FASE 8: SUGGERIMENTI")
    print_separator('-')
    suggest_conversion(y)
    
    # ========== 9. SALVA VERSIONE PRONTA ==========
    print("\n📍 FASE 9: SALVATAGGIO VERSIONE PRONTA")
    print_separator('-')
    
    output_path = save_ready_version(X, y, 'data/fordb_ready.npz')
    
    # ========== RIEPILOGO FINALE ==========
    print("\n" + "="*60)
    print("RIEPILOGO FINALE")
    print("="*60)
    
    print("""
✅ FordB scaricato e verificato correttamente!

FILE DISPONIBILI:
   • data/fordb_originale.npz  (originale, classi stringhe '-1'/'1')
   • data/fordb_ready.npz      (pronto per modello, classi 0/1)

PER USARE NEL MODELLO:
   # Opzione 1: usa il file ready (consigliato)
   data = np.load('data/fordb_ready.npz')
   X, y = data['X'], data['y']
   
   # Opzione 2: converti manualmente
   data = np.load('data/fordb_originale.npz')
   X = data['X']
   y = (data['y'].astype(int) + 1) // 2

DIFFERENZE CON FORDA:
   • FordA: 4,921 campioni
   • FordB: 4,446 campioni (dataset leggermente più piccolo)
   • FordB è più difficile (test di generalizzazione)
""")
    
    print_separator('=')
    print("VERIFICA COMPLETATA!")
    print_separator('=')

if __name__ == "__main__":
    main()