import numpy as np
import torch
from torch.utils.data import Dataset

class SupervisedEEGDataset(Dataset):
    def __init__(self, eeg_signals):
        if isinstance(eeg_signals, tuple) and len(eeg_signals) == 2:
            X, y = eeg_signals
            # Assicura che X sia 3D (campioni, canali, tempo)
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])
            elif len(X.shape) == 4:
                X = X.reshape(X.shape[0], X.shape[1], X.shape[3])
            self.data = X
            self.labels = y
        else:
            raise ValueError(f"Formato non supportato: {type(eeg_signals)}")
        
        print(f"SupervisedEEGDataset: data shape = {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Restituisce (signal, label) con signal shape (canali, tempo)
        signal = self.data[idx]  # Già shape (1, 500)
        label = self.labels[idx]
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
