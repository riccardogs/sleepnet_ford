import numpy as np
from sklearn.model_selection import train_test_split

class FordADataLoader:
    def __init__(self, data_path: str, labeled_fraction: float = 1.0):
        self.data_path = data_path
        self.labeled_fraction = labeled_fraction

    def load_data(self):
        data = np.load(self.data_path)
        X = data['X']
        y = data['y'].astype(int)
        y = (y + 1) // 2

        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        # 60% train, 20% val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )

        # TRAIN COMPLETO (per contrastive learning)
        X_train_all = X_train
        y_train_all = y_train

        # SOLO una % (o labeled_fraction) per il classificatore
        if self.labeled_fraction < 1.0:
            labeled_indices = []
            for label in np.unique(y_train):
                idx = np.where(y_train == label)[0]
                n = max(1, int(len(idx) * self.labeled_fraction))
                chosen = np.random.choice(idx, n, replace=False)
                labeled_indices.extend(chosen)

            X_train_labeled = X_train[labeled_indices]
            y_train_labeled = y_train[labeled_indices]
        else:
            X_train_labeled = X_train
            y_train_labeled = y_train

        return {
            'train_all': (X_train_all, y_train_all),           # per contrastive learning
            'train_labeled': (X_train_labeled, y_train_labeled), # per classifier
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }