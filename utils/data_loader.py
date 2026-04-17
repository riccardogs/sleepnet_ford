"""
FORD DATA LOADER - Adattato da EEG loader

1. STRUTTURA DEI DATI
   - EEG: File multipli .npz (uno per soggetto) in una directory
   - FordA: Singolo file .npz

2. FORMATO DEI FILE .NPZ
   - EEG: Chiavi 'x' (segnali) e 'y' (etichette)
   - FordA: Chiavi 'X' (segnali) e 'y' (etichette)

3. SHAPE DEI SEGNALI
   - EEG: (N_epochs, 1, 3000) - 3D (campioni, canali, tempo)
   - FordA: (N_campioni, 500) o (N_campioni, 1, 500) - 2D o 3D

4. NUMERO DI CLASSI
   - EEG: 5 classi (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
   - FordA: 2 classi (-1=normale, 1=anomalo) → convertite in (0,1)

5. SPLIT DEI DATI
   - EEG: Split per soggetti (85% train, 15% test) per evitare data leakage
   - FordA: Split randomico semplice (train/val/test) con stratificazione

6. IDENTIFICAZIONE SOGGETTI
   - EEG: Dai nomi dei file (es. "SC4001.npz" → soggetto 40)
   - FordA: Non applicabile (dati da sensori, non da soggetti)

7. COMPLESSITÀ DEL CODICE
   - EEG: ~150 righe (gestione file multipli, soggetti, split complesso)
   - FordA: ~30 righe (caricamento diretto, split semplice)

================================================================================
PERCHÉ QUESTE DIFFERENZE?
================================================================================

EEG (Sleep-EDF):
- Dataset medico con soggetti diversi
- Necessario split per soggetti per valutare generalizzazione
- Ogni soggetto ha più file notturni
- 5 stadi del sonno (task di classificazione)

FordA:
- Dataset industriale da sensori
- Tutti i campioni indipendenti (nessun concetto di "soggetto")
- Task binario (normale vs anomalo)
- Split randomico sufficiente

================================================================================
ADATTAMENTI NECESSARI PER FORDA
================================================================================

1. Cambiare num_classes da 5 a 2
2. Rimuovere logica di estrazione soggetti
3. Modificare split da "per soggetto" a "random stratificato"
4. Convertire etichette: (-1,1) → (0,1)
5. Gestire shape 2D (aggiungere dimensione canale se necessario)

================================================================================

"""



import numpy as np
from sklearn.model_selection import train_test_split

class FordADataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self):
        """Carica FordA nel formato richiesto da SimpleSleepNet"""
        
        # 1. Carica il file originale
        data = np.load(self.data_path)
        
        # 2. FordA ha struttura diversa da EEG
        #    Originale: X shape (4921, 1, 500), y shape (4921,)
        X = data['X']
        y = data['y'].astype(int)  # Converte '-1','1' in -1,1
        
        # 3. Rinomina classi: -1 → 0 (normale), 1 → 1 (anomalo)
        y = (y + 1) // 2
        
        # 4. Assicura shape 3D se necessario (campioni, canali, tempo)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # 5. Split originale di FordA (3601 train, 1320 test)
        #    Si riunisce e risuddivide train/val/test per il modello
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }