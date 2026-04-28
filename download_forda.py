from aeon.datasets import load_classification
import numpy as np

print(' Scaricando FordA originale con aeon...')


# per FordB basta mettere B

X, y = load_classification('FordA')

print(f'✅ Dataset caricato: {X.shape}')
print(f'   Shape: {X.shape}')
print(f'   y shape: {y.shape}')

# Salva originale senza modifiche
np.savez('data/forda_originale.npz',
         X=X,
         y=y)

print('💾 Salvato in data/forda_originale.npz')
print(f'   Campioni totali: {X.shape[0]}')
print(f'   Lunghezza sequenza: {X.shape[2]}')
print(f'   Canali: {X.shape[1]}')
print(f'   Classi: {np.unique(y)}')
