import random
import numpy as np
import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ContrastiveEEGDataset(Dataset):
    def __init__(self, eeg_signals, augmentations=None, return_labels=False):
        logger.debug(f"Creating ContrastiveEEGDataset")
        
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
            
        self.augmentations = augmentations or []
        self.return_labels = return_labels
        self.requires_x_random = any(getattr(aug, 'requires_x_random', False) for aug in self.augmentations)
        logger.info(f"ContrastiveEEGDataset created with {len(self.data)} samples, shape: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            signal = self.data[idx]  # Shape (1, 500)
            label = self.labels[idx]
            
            # Converti in 1D per le augmentations
            signal_1d = signal.squeeze(0)  # Shape (500,)
            
            random_signal = self.data[random.randint(0, len(self.data) - 1)] if self.requires_x_random else None
            random_signal_1d = random_signal.squeeze(0) if random_signal is not None else None
            
            augmented_signal_i = self.apply_augmentations(signal_1d.copy(), random_signal_1d)
            augmented_signal_j = self.apply_augmentations(signal_1d.copy(), random_signal_1d)
            
            # Converti in tensore e aggiungi dimensione canale
            augmented_signal_i = torch.tensor(augmented_signal_i, dtype=torch.float32).unsqueeze(0)
            augmented_signal_j = torch.tensor(augmented_signal_j, dtype=torch.float32).unsqueeze(0)
            
            if self.return_labels:
                label = torch.tensor(label, dtype=torch.long)
                return augmented_signal_i, augmented_signal_j, label
            else:
                return augmented_signal_i, augmented_signal_j
        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {e}")
            raise

    def apply_augmentations(self, signal, random_signal):
        for aug in self.augmentations:
            signal = aug(signal, random_signal) if getattr(aug, 'requires_x_random', False) else aug(signal)
        return signal
