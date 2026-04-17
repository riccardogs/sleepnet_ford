"""
SIMPLE SLEEP NET - ENCODER CNN 1D
==================================
Questo file definisce l'architettura dell'encoder che trasforma i segnali EEG
in embeddings vettoriali. Questi embeddings vengono poi usati:
    - Nel contrastive learning: per calcolare la similarità tra views
    - Nel classificatore: come features per predire gli stadi del sonno

Architettura:
    Input: segnale EEG grezzo (1 canale, 3000 campioni)
    ↓
    CNN 1D a 3 blocchi (convoluzioni dilate)
    ↓
    Global Average Pooling
    ↓
    MLP (Linear + BatchNorm + Mish)
    ↓
    Output: embedding normalizzato L2 (latent_dim dimensioni)

Caratteristiche:
    - Convoluzioni dilate: aumentano il receptive field senza perdere risoluzione
    - Mish activation: più smooth di ReLU, migliori gradienti
    - L2 normalization: embeddings su sfera unitaria (similarità coseno)
"""




"""
MODIFICATO


================================================================================
PERCHÉ QUESTE MODIFICHE NELL'ENCODER (simple_sleep_net.py)
================================================================================

IL PROBLEMA:
------------
L'encoder originale era progettato per INPUT DI 3000 CAMPIONI (EEG a 100Hz per 30 secondi).
FordA ha INPUT DI 500 CAMPIONI (~2 secondi a 250Hz).

Con i parametri originali (kernel_size=64, stride=8, padding=63), la convoluzione
cercava di elaborare un segnale di 500 campioni con filtri pensati per 3000 campioni,
causando:
- Padding eccessivo (63 campioni di zero su 500 totali)
- Kernel troppo grandi (64 campioni su 500)
- Stride troppo alti (riduce troppo velocemente la dimensione)

================================================================================
LE MODIFICHE PRINCIPALI
================================================================================

1. RIDUZIONE PROPORZIONALE DEI KERNEL
   - Prima: kernel_size=64 (2.1% di 3000)
   - Dopo:  kernel_size=16 (3.2% di 500)
   → Il kernel copre una porzione simile del segnale (2-3%)

2. RIDUZIONE PROPORZIONALE DELLO STRIDE
   - Prima: stride=8 (riduce dimensione di 8x)
   - Dopo:  stride=4 (riduce dimensione di 4x)
   → Per 500 campioni, stride=8 ridurrebbe troppo velocemente

3. RIDUZIONE PROPORZIONALE DEL PADDING
   - Prima: padding=63 (12.6% di 500 - assurdo!)
   - Dopo:  padding=7 (1.4% di 500 - ragionevole)
   → Il padding originale era enorme per FordA

4. RIDUZIONE DEL NUMERO DI FILTRI
   - Prima: 32 → 64 → 128 filtri
   - Dopo:  16 → 32 → 64 filtri
   → Dataset più piccolo (5000 campioni vs ~40000 EEG)
   → Riduce overfitting e parametri inutili

5. RIDUZIONE DEL LATENT_DIM
   - Prima: 128 dimensioni
   - Dopo:  64 dimensioni (parametro di default)
   → Spazio latente più compatto per dataset più piccolo

================================================================================
CONFRONTO DETTAGLIATO
================================================================================

BLOCCO 1 (primo layer convolutivo):
┌─────────────────┬──────────────────┬─────────────────┐
│ Parametro       │ Prima (EEG)      │ Dopo (FordA)    │
├─────────────────┼──────────────────┼─────────────────┤
│ in_channels     │ 1                │ 1               │
│ out_channels    │ 32               │ 16              │
│ kernel_size     │ 64               │ 16              │
│ stride          │ 8                │ 4               │
│ padding         │ 63               │ 7               │
│ dilation        │ 1                │ 1               │
└─────────────────┴──────────────────┴─────────────────┘

BLOCCO 2:
┌─────────────────┬──────────────────┬─────────────────┐
│ Parametro       │ Prima (EEG)      │ Dopo (FordA)    │
├─────────────────┼──────────────────┼─────────────────┤
│ in_channels     │ 32               │ 16              │
│ out_channels    │ 64               │ 32              │
│ kernel_size     │ 32               │ 8               │
│ stride          │ 4                │ 2               │
│ padding         │ 62               │ 5               │
│ dilation        │ 2                │ 2               │
└─────────────────┴──────────────────┴─────────────────┘

BLOCCO 3:
┌─────────────────┬──────────────────┬─────────────────┐
│ Parametro       │ Prima (EEG)      │ Dopo (FordA)    │
├─────────────────┼──────────────────┼─────────────────┤
│ in_channels     │ 64               │ 32              │
│ out_channels    │ 128              │ 64              │
│ kernel_size     │ 16               │ 4               │
│ stride          │ 2                │ 2               │
│ padding         │ 60               │ 4               │
│ dilation        │ 4                │ 4               │
└─────────────────┴──────────────────┴─────────────────┘

================================================================================
IMPATTO SULLE DIMENSIONI
================================================================================

PERCORSO DEL SEGNALE (da 500 campioni a embedding):

Input:                    (Batch, 1, 500)
                           ↓
Conv1 (k16, s4, pad7):    (Batch, 16, ~125)   # 500/4 ≈ 125
                           ↓
Conv2 (k8, s2, pad5):     (Batch, 32, ~62)    # 125/2 ≈ 62
                           ↓
Conv3 (k4, s2, pad4):     (Batch, 64, ~31)    # 62/2 ≈ 31
                           ↓
AdaptiveAvgPool1d(1):     (Batch, 64, 1)
                           ↓
Flatten:                  (Batch, 64)
                           ↓
Linear(64 → latent_dim):  (Batch, latent_dim)  # default 64
                           ↓
L2 Normalization:         (Batch, latent_dim)

================================================================================
PERCHÉ QUESTE MODIFICHE SONO CORRETTE?
================================================================================

1. PRESERVANO LA GERARCHIA TEMPORALE
   - 3 blocchi convolutivi come nell'originale
   - Convoluzioni dilate mantenute (dilation=1,2,4)
   - Stessa filosofia architetturale

2. RISPETTANO LE PROPORZIONI
   - Il rapporto kernel_size/input_length è simile (~2-3%)
   - Il numero di layer è lo stesso (3 blocchi)
   - La riduzione dimensionale è graduale

3. EVITANO OVERFITTING
   - Meno parametri totali (adatto a dataset più piccolo)
   - Dropout mantenuto a 0.2
   - BatchNorm presente in ogni blocco

4. MANTENGONO LA FUNZIONALITÀ
   - L2 normalization alla fine (per similarità coseno)
   - Projection head (MLP finale)
   - AdaptiveAvgPooling (indipendente dalla lunghezza)

================================================================================
RISULTATO: L'ENCODER ORA FUNZIONA CORRETTAMENTE CON FORDA (500 CAMPIONI)!
================================================================================


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MISH ACTIVATION FUNCTION
# ============================================================================
# La Mish è una funzione di attivazione proposta nel paper "Mish: A Self Regularized
# Non-Monotonic Activation Function" (Diganta Misra, 2019).
#
# Formula: Mish(x) = x * tanh(softplus(x))
# dove softplus(x) = log(1 + e^x)
#
# Vantaggi rispetto a ReLU:
#   1. Non monotona (permette gradienti negativi piccoli)
#   2. Smooth (derivata continua, miglior training)
#   3. Self-regularizing (leggermente)
#   4. Spesso outperforma ReLU su task complessi
#
# Grafico: forma a "s" morbida, valori negativi piccoli vicino a zero
# ============================================================================
class Mish(nn.Module):
    """
    Mish Activation Function: x * tanh(softplus(x))
    
    Softplus(x) = log(1 + e^x) è una approssimazione smooth di ReLU.
    """
    def forward(self, x):
        # F.softplus(x) = log(1 + exp(x))
        # torch.tanh() = (e^x - e^{-x}) / (e^x + e^{-x})
        return x * torch.tanh(F.softplus(x))


# ============================================================================
# SIMPLE SLEEP NET - ENCODER
# ============================================================================
class SimpleSleepNet(nn.Module):
    """
    SimpleSleepNet: Encoder CNN 1D per segnali EEG
    
    Architettura dettagliata:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Input: (Batch, 1 canale, 3000 campioni)                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Conv1d: 1→32 canali, kernel=64, stride=8, padding=63, dilation=1   │
    │ BatchNorm1d(32) + Mish() + Dropout(0.2)                            │
    │   → Dopo conv: (Batch, 32, ~375)                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Conv1d: 32→64 canali, kernel=32, stride=4, padding=62, dilation=2  │
    │ BatchNorm1d(64) + Mish() + Dropout(0.2)                            │
    │   → Dopo conv: (Batch, 64, ~94)                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Conv1d: 64→128 canali, kernel=16, stride=2, padding=60, dilation=4 │
    │ BatchNorm1d(128) + Mish() + Dropout(0.2)                           │
    │   → Dopo conv: (Batch, 128, ~47)                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ AdaptiveAvgPool1d(1) → (Batch, 128, 1)                             │
    │ Flatten → (Batch, 128)                                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Linear(128 → latent_dim)                                            │
    │ BatchNorm1d(latent_dim) + Mish() + Dropout(0.2)                    │
    │   → Output: (Batch, latent_dim)                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ L2 Normalization (proietta embeddings sulla sfera unitaria)        │
    └─────────────────────────────────────────────────────────────────────┘
    
    Perché convoluzioni dilate?
        - Dilated convolution: salta 'dilation-1' campioni tra input e kernel
        - Aumenta il receptive field senza aumentare il numero di parametri
        - Permette di catturare pattern a diverse scale temporali
        
    Parametri:
    ----------
    latent_dim : int
        Dimensione dello spazio latente (embedding finale). Default: 128
    dropout : float
        Tasso di dropout (probabilità di azzerare un neurone). Default: 0.2
    """
    
    def __init__(self, latent_dim=128, dropout=0.2):
        super(SimpleSleepNet, self).__init__()
        
        logger.info(f"Initializing SimpleSleepNet with latent_dim={latent_dim} and dropout={dropout}")
        
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=dropout)

        """
        # ====================================================================
        # CONVOLUTIONAL PATH (Feature Extractor)
        # ====================================================================
        # 3 blocchi convolutivi con:
        #   - Conv1d: estrae pattern temporali
        #   - BatchNorm: normalizza per stabilizzare il training
        #   - Mish: attivazione non lineare
        #   - Dropout: regolarizzazione (previene overfitting)
        # ====================================================================
        self.conv_path = nn.Sequential(
            # ---------- BLOCCO 1 ----------
            # Input: (B, 1, 3000)
            # Output: (B, 32, ~375)
            nn.Conv1d(
                in_channels=1,      # EEG è mono-canale (Fpz-Cz)
                out_channels=32,    # Numero di filtri (feature maps)
                kernel_size=64,     # Grandezza del filtro (cattura ~64 campioni)
                stride=8,           # Quanto spostare il filtro (riduce dimensione)
                padding=63,         # Mantiene la dimensione? No, stride la riduce
                dilation=1,         # Dilation base (nessun salto)
                bias=False          # BatchNorm già ha un bias
            ),
            nn.BatchNorm1d(32),     # Normalizza per canale (media=0, var=1)
            Mish(),                  # Attivazione non lineare
            self.dropout,           # Regolarizzazione
            
            # ---------- BLOCCO 2 ----------
            # Input: (B, 32, ~375)
            # Output: (B, 64, ~94)
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=32,
                stride=4,
                padding=62,
                dilation=2,         # Salta 1 campione tra input (receptive field più largo)
                bias=False
            ),
            nn.BatchNorm1d(64),
            Mish(),
            self.dropout,
            
            # ---------- BLOCCO 3 ----------
            # Input: (B, 64, ~94)
            # Output: (B, 128, ~47)
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=16,
                stride=2,
                padding=60,
                dilation=4,         # Salta 3 campioni tra input (receptive field molto largo)
                bias=False
            ),
            nn.BatchNorm1d(128),
            Mish(),
            self.dropout,
        )
        """

        self.conv_path = nn.Sequential(
            # BLOCCO 1: 500 → ~125 campioni
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=16,
                stride=4,
                padding=7,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm1d(16),
            Mish(),
            self.dropout,
            
            # BLOCCO 2: ~125 → ~62 campioni
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=8,
                stride=2,
                padding=5,
                dilation=2,
                bias=False
            ),
            nn.BatchNorm1d(32),
            Mish(),
            self.dropout,
            
            # BLOCCO 3: ~62 → ~31 campioni
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=4,
                dilation=4,
                bias=False
            ),
            nn.BatchNorm1d(64),
            Mish(),
            self.dropout,
        )
        

    
        # ====================================================================
        # FULLY CONNECTED PATH (Projection Head)
        # ====================================================================
        # Riduce la dimensione da 128 a latent_dim (es. 128 → 64 o 128)
        # Questo strato è talvolta chiamato "projection head" nel contrastive learning
        # ====================================================================
        self.fc = nn.Sequential(
            nn.Linear(64, self.latent_dim),        # Proiezione lineare
            nn.BatchNorm1d(self.latent_dim),        # Normalizzazione
            Mish(),                                 # Attivazione
            self.dropout                            # Regolarizzazione
        )
        
        logger.info("SimpleSleepNet initialization complete.")
    
    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    def forward(self, x):
        """
        Forward pass del modello.
        
        Input:
            x : torch.Tensor
                Segnale EEG grezzo di shape (Batch, Channels, Length)
                Esempio: (256, 1, 3000) dove 3000 campioni ~30 secondi a 100Hz
        
        Output:
            x : torch.Tensor
                Embedding normalizzato L2 di shape (Batch, latent_dim)
                Esempio: (256, 128) o (256, 64)
                
        Processo:
            1. Estrae features con la CNN 1D
            2. Global average pooling: media su tutta la dimensione temporale
            3. Flatten: (B, C, 1) → (B, C)
            4. Proietta nello spazio latente con MLP
            5. L2 normalization: ||output|| = 1 (proiezione sulla sfera unitaria)
        """
        logger.debug("Starting forward pass.")
        
        # STEP 1: Estrazione features con CNN
        # Input: (B, 1, 3000) → Output: (B, 128, ~47)
        x = self.conv_path(x)
        
        # STEP 2: Global Average Pooling
        # Prende la media su tutta la dimensione temporale (ultima dimensione)
        # Output: (B, 128, 1)
        x = F.adaptive_avg_pool1d(x, 1)
        
        # STEP 3: Flatten
        # Rimuove la dimensione singleton (1) → (B, 128)
        x = x.view(x.size(0), -1)
        
        # STEP 4: Proiezione nello spazio latente
        # (B, 128) → (B, latent_dim)
        x = self.fc(x)
        
        # STEP 5: L2 Normalization
        # Proietta gli embeddings sulla sfera unitaria
        # Questo è fondamentale per il contrastive learning:
        #   - La similarità coseno = prodotto scalare dopo L2-norm
        #   - Tutti gli embeddings hanno la stessa magnitudine
        #   - Previene il collapse delle rappresentazioni
        x = F.normalize(x, p=2, dim=1)
        
        logger.debug("Forward pass complete.")
        return x


# ============================================================================
# NOTE SULLE DIMENSIONI (con batch_size=256, latent_dim=128)
# ============================================================================
#
# Input:           (256, 1, 3000)
#                  
# Dopo Conv1 (64, stride8, pad63, dil1):
#   L_out = floor((3000 + 2*63 - 64*(1) - (64-1)*(1-1)) / 8) + 1
#         = floor((3000 + 126 - 64) / 8) + 1
#         = floor(3062 / 8) + 1 = 382 + 1 = 383? (circa)
#   Output: (256, 32, ~375)
#
# Dopo Conv2 (32→64, k32, s4, pad62, dil2):
#   Output: (256, 64, ~94)
#
# Dopo Conv3 (64→128, k16, s2, pad60, dil4):
#   Output: (256, 128, ~47)
#
# Dopo AdaptiveAvgPool1d(1):
#   Output: (256, 128, 1)
#
# Dopo Flatten:
#   Output: (256, 128)
#
# Dopo Linear(128→128):
#   Output: (256, 128)
#
# Dopo L2 Normalization:
#   Output: (256, 128) con ||output|| = 1 per ogni riga
# ============================================================================