##---------------------------------------------------------------------
# File: pca_ica.py
# Author: Karthik Singaravadivelan, Anant Gupta
# Description: ZCA whitening model for embedding normalization.
##----------------------------------------------------------------------
import numpy as np
import pickle

class ZCAWhiteningModel:
    def __init__(self, mean: np.ndarray, whitening_matrix: np.ndarray, eps: float = 1e-8):
        self.mean = mean
        self.whitening_matrix = whitening_matrix
        self.eps = eps

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ZCA whitening to a single embedding or a batch.
        """
        is_single = (x.ndim == 1)
        if is_single:
            x = x[np.newaxis, :]

        # Center
        x_centered = x - self.mean

        # Apply ZCA whitening matrix
        x_whitened = np.dot(x_centered, self.whitening_matrix.T)

        return x_whitened[0] if is_single else x_whitened

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8):
        """
        Fit ZCA whitening on embedding matrix X.
        """
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Compute covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Eigen-decomposition of covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Add epsilon for numerical stability
        eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals + eps)

        # Compute whitening matrix
        whitening_matrix = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

        return cls(mean, whitening_matrix, eps)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'whitening_matrix': self.whitening_matrix,
                'eps': self.eps
            }, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(
            mean=data['mean'],
            whitening_matrix=data['whitening_matrix'],
            eps=data['eps']
        )

def encode_and_whiten_zca(obj) -> np.ndarray:
    """
    Encode sentences with SentenceTransformer and whiten embeddings with ZCA whitening model.

    Applies whitening if a numpy array is passed in, hence general keywording!
    """
    if type(obj[0]) == str:
        # Step 1: Encode sentences to embeddings (numpy array)
        embeddings = st_model.encode(obj, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    else:
        embeddings = obj

    # Step 2: Apply ZCA whitening transform
    whitened_embeddings = zca_whitening_model.transform(embeddings)

    return whitened_embeddings
