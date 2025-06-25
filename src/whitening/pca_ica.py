##---------------------------------------------------------------------
# File: pca_ica.py
# Author: Karthik Singaravadivelan, Anant Gupta
# Description: PCA + ICA whitening model for embedding normalization.
##----------------------------------------------------------------------
import numpy as np
import pickle
from sklearn.decomposition import PCA, FastICA

class PCAICAWhitening:
    def __init__(self, pca_dim=256, eps=1e-8, ica_max_iter=5000, ica_tol=1e-3):
        self.pca_dim = pca_dim
        self.eps = eps
        self.ica_max_iter = ica_max_iter
        self.ica_tol = ica_tol

        # Trained components
        self.mean = None
        self.pca_components = None
        self.pca_var = None
        self.ica_unmixing = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # PCA
        pca = PCA(n_components=self.pca_dim)
        X_pca = pca.fit_transform(X_centered)
        self.pca_components = pca.components_
        self.pca_var = pca.explained_variance_

        # Normalize PCA output
        X_pca_norm = X_pca / np.sqrt(self.pca_var + self.eps)

        # ICA
        ica = FastICA(n_components=self.pca_dim, whiten='unit-variance',
                      max_iter=self.ica_max_iter, tol=self.ica_tol, random_state=42)
        X_ica = ica.fit_transform(X_pca_norm)
        self.ica_unmixing = ica.components_
        return self

    def transform(self, X):
        if self.mean is None:
            raise ValueError("Model not fitted yet.")

        X = np.atleast_2d(X)
        X_centered = X - self.mean
        X_pca = (X_centered @ self.pca_components.T) / np.sqrt(self.pca_var + self.eps)
        X_ica = X_pca @ self.ica_unmixing.T
        return X_ica[0] if X.shape[0] == 1 else X_ica

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls()
        obj.__dict__.update(state)
        return obj


def encode_and_whiten_pcaica(sentences, st_model, whitening_model) -> np.ndarray:
    """
    Encode sentences with SentenceTransformer and whiten embeddings with PCA whitening model.

    Args:
        sentences (list of str or np.ndarray): List of sentences or numpy array of embeddings.
        st_model (SentenceTransformer): An initialized SentenceTransformer model.
        whitening_model (PCAICAWhiteningModel): An initialized PCAICAWhiteningModel.

    Returns:
        np.ndarray: Whitened embeddings.
    """
    if isinstance(sentences[0], str):
        # Step 1: Encode sentences to embeddings (numpy array)
        embeddings = st_model.encode(sentences, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    else:
        embeddings = sentences

    # Step 2: Apply whitening transform
    whitened_embeddings = whitening_model.transform(embeddings)

    return whitened_embeddings