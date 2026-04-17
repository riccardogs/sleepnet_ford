from aeon.datasets import load_classification
import numpy as np
from sklearn.model_selection import train_test_split

print('📥 Scaricando vero FordA...')

# Carica il dataset FordA
X, y = load_classification('FordA')

print(f'✅ Dataset caricato: {X.shape}')
print(f'   Shape X: {X.shape}')
print(f'   Shape y: {y.shape}')
print(f'   Tipo y: {type(y)}')
print(f'   Valori unici in y: {np.unique(y)}')

# Se y è un array di stringhe, converti in numeri
if y.dtype == object or isinstance(y[0], str):
    print('   Convertendo etichette da stringhe a numeri...')
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[val] for val in y])
    print(f'   Mappa: {label_map}')
    print(f'   Nuovi valori unici: {np.unique(y)}')

# X è 3D: (campioni, canali, tempo) -> (4921, 1, 500)
# Converti in 2D: (campioni, tempo)
X_2d = X.reshape(X.shape[0], -1)

print(f'\n📊 Dataset completo: {X_2d.shape[0]} campioni')
print(f'   Classe 0: {sum(y==0)}')
print(f'   Classe 1: {sum(y==1)}')

# Split in train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(X_2d, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Salva
np.savez('data/forda_real.npz',
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)

print(f'\n💾 Salvato in data/forda_real.npz')
print(f'   Train: {X_train.shape} (classi: 0={sum(y_train==0)}, 1={sum(y_train==1)})')
print(f'   Val:   {X_val.shape} (classi: 0={sum(y_val==0)}, 1={sum(y_val==1)})')
print(f'   Test:  {X_test.shape} (classi: 0={sum(y_test==0)}, 1={sum(y_test==1)})')
