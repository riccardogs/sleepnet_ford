"""
FORD DATA LOADER - Solo per FordA
"""

import numpy as np
from sklearn.model_selection import train_test_split

class FordADataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self):
        """Carica FordA nel formato richiesto da SimpleSleepNet"""
        
        # Carica il file originale
        data = np.load(self.data_path)
        
        # FordA: X shape (4921, 1, 500), y shape (4921,)
        X = data['X']
        y = data['y'].astype(int)
        
        # Rinomina classi: -1 → 0 (normale), 1 → 1 (anomalo)
        y = (y + 1) // 2
        
        # Assicura shape 3D (campioni, canali, tempo)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Split in train/val/test
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
