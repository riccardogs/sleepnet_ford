import pandas as pd
import numpy as np
import os

print("📥 Download FordA dataset...")

# URL del dataset
url_train = 'https://www.timeseriesclassification.com/Downloads/FordA_TRAIN.tsv'
url_test = 'https://www.timeseriesclassification.com/Downloads/FordA_TEST.tsv'

# Scarica
train = pd.read_csv(url_train, sep='\t', header=None)
test = pd.read_csv(url_test, sep='\t', header=None)

print(f"✅ Train: {train.shape}")
print(f"✅ Test: {test.shape}")

# Salva
np.save('data/forda_train.npy', train.values)
np.save('data/forda_test.npy', test.values)

print("💾 Dataset salvato in data/")
print(f"   - Classi nel train: {np.unique(train.iloc[:, 0].values)}")
print(f"   - Classi nel test: {np.unique(test.iloc[:, 0].values)}")
