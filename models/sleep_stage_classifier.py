"""
SLEEP STAGE CLASSIFIER - MLP PER CLASSIFICAZIONE
=================================================
Questo file definisce il classificatore MLP che trasforma gli embeddings
prodotti dall'encoder SimpleSleepNet in predizioni degli stadi del sonno.

MOFIFICATO SOLO: num_classes: int = 2

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MISH ACTIVATION FUNCTION
# ============================================================================
# Riutilizziamo la stessa Mish definita nell'encoder
# Formula: Mish(x) = x * tanh(softplus(x))
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ============================================================================
# SLEEP STAGE CLASSIFIER - MLP
# ============================================================================
class SleepStageClassifier(nn.Module):
    
    def __init__(self, input_dim: int = 128, num_classes: int = 2, dropout_probs: float = 0.5):
        super(SleepStageClassifier, self).__init__()
        
        # Validazione del tipo di dropout_probs
        if not isinstance(dropout_probs, float):
            raise ValueError("dropout_probs must be a float.")
        
        logger.info(f"Initializing SleepStageClassifier with input_dim={input_dim}, "
                   f"num_classes={num_classes}, dropout_probs={dropout_probs}")
        
        # ====================================================================
        # CLASSIFIER MLP
        # ====================================================================
        # Un MLP a 3 layer (2 nascosti, 1 output)
        # BatchNorm: normalizza l'input di ogni layer (media=0, var=1)
        # Mish: attivazione non lineare (alternativa a ReLU)
        # Dropout: regolarizzazione (previene overfitting)
        # ====================================================================
        self.classifier = nn.Sequential(
            # ---------- LAYER 1: Espansione ----------
            # input_dim → 256 (espande le features per maggiore capacità)
            nn.Linear(input_dim, 256),
            # BatchNorm: stabilizza il training, permette learning rate più alti
            nn.BatchNorm1d(256),
            # Mish: attivazione smooth (migliore di ReLU per gradienti)
            Mish(),
            # Dropout: disattiva casualmente il 50% dei neuroni durante training
            nn.Dropout(p=dropout_probs),
            
            # ---------- LAYER 2: Compressione ----------
            # 256 → 128 (comprime le features)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            Mish(),
            nn.Dropout(p=dropout_probs),
            
            # ---------- LAYER 3: Output ----------
            # 128 → 5 (logits per le 5 classi)
            # NOTA: Non usiamo softmax qui perché CrossEntropyLoss la applica internamente
            nn.Linear(128, num_classes)
        )
        
        logger.debug("SleepStageClassifier model architecture created.")
    
    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        logger.debug("Forward pass started.")
        
        # Passa attraverso tutti i layer del MLP
        x = self.classifier(x)
        
        logger.debug("Forward pass completed.")
        
        # Restituisce logits (non softmax)
        # La loss CrossEntropyLoss applicherà softmax internamente
        return x

