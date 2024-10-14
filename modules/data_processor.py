import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataProcessor:
    def __init__(self, pca_enable=True, pca_components=2):
        self.pca_enable = pca_enable
        self.pca = PCA(n_components=pca_components)
        self.encoder = OneHotEncoder(sparse_output =False)

    def load_data(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = np.loadtxt(file)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None, None, None, None
        except Exception as e:
            print(f"Data loading error: {str(e)}")
            return None, None, None, None
        
        X, y = self._split_features_and_labels(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        if self.pca_enable:
            X_train, X_test = self._apply_pca(X_train, X_test)

        return X_train, y_train, X_test, y_test

    def _split_features_and_labels(self, dataset):
        X = dataset[:, :-1]
        y = self.encoder.fit_transform(dataset[:, -1].astype(int).reshape(-1, 1))
        return X, y

    def _apply_pca(self, X_train, X_test):
        self.pca.fit(X_train)
        X_train = self.pca.transform(X_train)
        X_test = self.pca.transform(X_test)
        return X_train, X_test
