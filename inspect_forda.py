"""
ANALISI STRUTTURA DATASET FORDA
"""

import numpy as np

DATA_PATH = "/Users/riccardosasu/Desktop/sleepnet_ford/data/forda_real.npz"

data = np.load(DATA_PATH)

print("=" * 80)
print("STRUTTURA DATASET FORDA")
print("=" * 80)

print("\n📁 CHIAVI NEL FILE:")
for key in data.keys():
    print(f"   - {key}")

print("\n" + "=" * 80)
print("DETTAGLI PER CHIAVE")
print("=" * 80)

for key in data.keys():
    arr = data[key]
    print(f"\n🔍 {key}:")
    print(f"   Shape: {arr.shape}")
    print(f"   Tipo: {arr.dtype}")
    print(f"   Dimensione in memoria: {arr.nbytes / 1024:.2f} KB")
    
    if len(arr.shape) == 1:
        print(f"   Valori unici: {np.unique(arr)}")
        print(f"   Prime 10 etichette: {arr[:10]}")
    else:
        print(f"   Min: {arr.min():.4f}")
        print(f"   Max: {arr.max():.4f}")
        print(f"   Media: {arr.mean():.4f}")
        print(f"   Std: {arr.std():.4f}")
        print(f"   Prime 2 righe (prime 10 colonne):\n{arr[:2, :10]}")

print("\n" + "=" * 80)
print("INTERPRETAZIONE")
print("=" * 80)

X_train = data['X_train']
print(f"\n📊 FORMATO DATI:")
print(f"   Shape X_train: (campioni, tempo) = {X_train.shape}")
print(f"   → {X_train.shape[0]} campioni di training")
print(f"   → {X_train.shape[1]} punti temporali per campione")
print(f"   → 1 canale (monodimensionale)")

print(f"\n🎯 FORMATO ETICHETTE:")
print(f"   Shape y_train: (campioni,) = {data['y_train'].shape}")
print(f"   Classi: {np.unique(data['y_train'])} (0=normale, 1=anomalo)")

print(f"\n⏱️  FREQUENZA CAMPIONAMENTO:")
n_samples = X_train.shape[1]
print(f"   La lunghezza di {n_samples} punti corrisponde a circa 1-2 secondi")
print(f"   Frequenza stimata: {n_samples / 1.5:.0f}-{n_samples / 1:.0f} Hz")

print("\n✅ ANALISI COMPLETATA")