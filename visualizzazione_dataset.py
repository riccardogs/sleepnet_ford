"""
VISUALIZZATORE DATASET FORDA 
=================================================
Questo script permette di:
1. Visualizzare le shape del dataset FordA
2. Plottare segnali normali vs anomali
3. Mostrare la distribuzione delle etichette
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DATA_PATH = "/Users/riccardosasu/Desktop/sleepnet_ford/data/forda_real.npz"

# Mappa etichette (FordA: 0=normale, 1=anomalo)
LABELS = {0: "NORMALE (Healthy)", 1: "ANOMALO (Fault)"}
COLORS = {0: "green", 1: "red"}

# Frequenza di campionamento (stimata per FordA - 500 punti in ~1-2 secondi)
SAMPLE_RATE = 250  # Hz (approssimativo)


# ============================================================================
# FUNZIONI DI CARICAMENTO
# ============================================================================

def load_forda_data(data_path=DATA_PATH):
    """Carica il dataset FordA dal file .npz."""
    data = np.load(data_path)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Combina train e val per analisi completa
    X = np.vstack([X_train, X_val])
    y = np.hstack([y_train, y_val])
    
    print("=" * 80)
    print("DATASET FORDA - CARICATO")
    print("=" * 80)
    print(f"\n📊 Shape dei dati:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Totale (train+val): {X.shape}")
    
    print(f"\n📊 Distribuzione etichette:")
    print(f"   Normali (0): {sum(y==0)}")
    print(f"   Anomali (1): {sum(y==1)}")
    
    return X, y, X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# FUNZIONI PER IL PLOT
# ============================================================================

def plot_signal(signal, label, title="FordA Signal"):
    """
    Plotta un singolo segnale FordA.
    """
    n_samples = len(signal)
    duration = n_samples / SAMPLE_RATE
    time_axis = np.linspace(0, duration, n_samples)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    color = COLORS.get(label, "blue")
    ax.plot(time_axis, signal, color=color, linewidth=0.8)
    ax.set_xlabel('Tempo (secondi)')
    ax.set_ylabel('Ampiezza (unità normalizzate)')
    ax.set_title(f'{title} - {LABELS.get(label, label)}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Statistiche segnale:")
    print(f"   Min: {signal.min():.4f}")
    print(f"   Max: {signal.max():.4f}")
    print(f"   Media: {signal.mean():.4f}")
    print(f"   Std: {signal.std():.4f}")
    print(f"   Durata: {duration:.2f} secondi")
    print(f"   Campioni: {n_samples}")
    print(f"   Frequenza stimata: {n_samples/duration:.1f} Hz")


def plot_label_distribution(y):
    """Plotta la distribuzione delle etichette."""
    label_counts = Counter(y)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels_names = [LABELS[i] for i in sorted(label_counts.keys())]
    counts = [label_counts[i] for i in sorted(label_counts.keys())]
    colors_list = [COLORS[i] for i in sorted(label_counts.keys())]
    
    bars = ax.bar(labels_names, counts, color=colors_list, edgecolor='black')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Numero di campioni')
    ax.set_title('Distribuzione delle classi nel dataset FordA')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Distribuzione classi:")
    total = sum(counts)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percent = count / total * 100
        print(f"   {LABELS[label]}: {count} campioni ({percent:.1f}%)")


def plot_normal_vs_anomaly(X, y):
    """Confronta un segnale normale con uno anomalo."""
    # Trova un indice normale e uno anomalo
    normal_idx = np.where(y == 0)[0][0]
    anomaly_idx = np.where(y == 1)[0][0]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    n_samples = X.shape[1]
    duration = n_samples / SAMPLE_RATE
    time_axis = np.linspace(0, duration, n_samples)
    
    # Segnale normale
    axes[0].plot(time_axis, X[normal_idx], 'green', linewidth=0.8)
    axes[0].set_ylabel('Ampiezza')
    axes[0].set_title(f'NORMALE - {LABELS[0]}')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, duration)
    
    # Segnale anomalo
    axes[1].plot(time_axis, X[anomaly_idx], 'red', linewidth=0.8)
    axes[1].set_xlabel('Tempo (secondi)')
    axes[1].set_ylabel('Ampiezza')
    axes[1].set_title(f'ANOMALO - {LABELS[1]}')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, duration)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Confronto statistico:")
    print(f"\nNORMALE:")
    print(f"   Min: {X[normal_idx].min():.4f}")
    print(f"   Max: {X[normal_idx].max():.4f}")
    print(f"   Std: {X[normal_idx].std():.4f}")
    print(f"\nANOMALO:")
    print(f"   Min: {X[anomaly_idx].min():.4f}")
    print(f"   Max: {X[anomaly_idx].max():.4f}")
    print(f"   Std: {X[anomaly_idx].std():.4f}")


def plot_multiple_signals(X, y, indices, title="Segnali FordA"):
    """Plotta più segnali in una figura."""
    n_plots = len(indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    n_samples = X.shape[1]
    duration = n_samples / SAMPLE_RATE
    time_axis = np.linspace(0, duration, n_samples)
    
    for i, idx in enumerate(indices):
        color = COLORS.get(y[idx], "blue")
        axes[i].plot(time_axis, X[idx], color=color, linewidth=0.8)
        axes[i].set_ylabel('Ampiezza')
        axes[i].set_title(f'Campione {idx} - {LABELS.get(y[idx], y[idx])}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, duration)
    
    axes[-1].set_xlabel('Tempo (secondi)')
    plt.tight_layout()
    plt.show()


# ============================================================================
# ANALISI STATISTICA
# ============================================================================

def print_statistics(X, y):
    """Stampa statistiche dettagliate del dataset."""
    print("\n" + "=" * 80)
    print("STATISTICHE DEL DATASET")
    print("=" * 80)
    
    print(f"\n📊 Dimensioni:")
    print(f"   Numero campioni: {X.shape[0]}")
    print(f"   Lunghezza sequenza: {X.shape[1]} punti")
    
    print(f"\n📊 Statistiche globali:")
    print(f"   Media globale: {X.mean():.4f}")
    print(f"   Std globale: {X.std():.4f}")
    print(f"   Min globale: {X.min():.4f}")
    print(f"   Max globale: {X.max():.4f}")
    
    print(f"\n📊 Statistiche per classe:")
    for label in [0, 1]:
        mask = y == label
        class_data = X[mask]
        print(f"\n   {LABELS[label]}:")
        print(f"      Media: {class_data.mean():.4f}")
        print(f"      Std: {class_data.std():.4f}")
        print(f"      Min: {class_data.min():.4f}")
        print(f"      Max: {class_data.max():.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Esegue l'analisi completa del dataset FordA."""
    
    print("=" * 80)
    print("VISUALIZZATORE DATASET FORDA")
    print("=" * 80)
    
    # 1. Carica il dataset
    print("\n🔍 CARICAMENTO DATASET...")
    X, y, X_train, y_train, X_val, y_val, X_test, y_test = load_forda_data()
    
    # 2. Distribuzione etichette
    print("\n" + "=" * 80)
    print("DISTRIBUZIONE CLASSI")
    print("=" * 80)
    plot_label_distribution(y)
    
    # 3. Statistiche dettagliate
    print_statistics(X, y)
    
    # 4. Confronto normale vs anomalo
    print("\n" + "=" * 80)
    print("CONFRONTO NORMALE vs ANOMALO")
    print("=" * 80)
    plot_normal_vs_anomaly(X, y)
    
    # 5. Esempi di segnali normali
    print("\n" + "=" * 80)
    print("ESEMPI DI SEGNALI NORMALI")
    print("=" * 80)
    normal_indices = np.where(y == 0)[0][:3]
    plot_multiple_signals(X, y, normal_indices, title="Segnali Normali")
    
    # 6. Esempi di segnali anomali
    print("\n" + "=" * 80)
    print("ESEMPI DI SEGNALI ANOMALI")
    print("=" * 80)
    anomaly_indices = np.where(y == 1)[0][:3]
    plot_multiple_signals(X, y, anomaly_indices, title="Segnali Anomali")
    
    # 7. Plot di un singolo segnale a scelta
    print("\n" + "=" * 80)
    print("ANALISI SINGOLO SEGNALE")
    print("=" * 80)
    sample_idx = 0
    plot_signal(X[sample_idx], y[sample_idx], title=f"FordA - Campione {sample_idx}")
    
    print("\n✅ Analisi completata!")


if __name__ == "__main__":
    main()