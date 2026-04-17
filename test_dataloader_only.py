"""
Test indipendente del FordA Dataloader
Non dipende da altri moduli del progetto
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
        
        # 5. Split in train/val/test
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


if __name__ == "__main__":
    print("=" * 60)
    print("TEST INDIPENDENTE DATALOADER FORDA")
    print("=" * 60)
    
    # Carica
    loader = FordADataLoader('data/forda_originale.npz')
    data = loader.load_data()
    
    # Verifica
    print("\n📊 RISULTATI:")
    print(f"   Train X: {data['train'][0].shape}")
    print(f"   Train y: {data['train'][1].shape}")
    print(f"   Val X:   {data['val'][0].shape}")
    print(f"   Val y:   {data['val'][1].shape}")
    print(f"   Test X:  {data['test'][0].shape}")
    print(f"   Test y:  {data['test'][1].shape}")
    
    # Verifica classi
    train_classes = np.unique(data['train'][1])
    print(f"\n🎯 Classi nel train: {train_classes}")
    print(f"   0 (normale): {sum(data['train'][1]==0)} campioni")
    print(f"   1 (anomalo): {sum(data['train'][1]==1)} campioni")
    
    # Verifica che non ci siano errori
    print("\n✅ DATALOADER FUNZIONANTE!")
    print("   (indipendente dal resto del codice)")
