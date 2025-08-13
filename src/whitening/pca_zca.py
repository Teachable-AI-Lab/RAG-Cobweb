import numpy as np
import pickle
from sklearn.decomposition import PCA

class PCAZCAWhiteningModel:
    def __init__(self, mean: np.ndarray, pca_components: np.ndarray,
                 pca_explained_var: np.ndarray, eps: float = 1e-8):
        self.mean = mean
        self.pca_components = pca_components
        self.pca_explained_var = pca_explained_var
        self.eps = eps

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  mean.shape={self.mean.shape},\n"
            f"  pca_components.shape={self.pca_components.shape},\n"
            f"  pca_explained_var.shape={self.pca_explained_var.shape},\n"
            f"  eps={self.eps}\n"
            f")"
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply PCA + ZCA whitening to a single embedding or a batch.
        """
        is_single = (x.ndim == 1)
        if is_single:
            x = x[np.newaxis, :]

        # Step 1: Center
        x_centered = x - self.mean

        # Step 2: PCA projection
        x_pca = np.dot(x_centered, self.pca_components.T)

        # Step 3: Normalize by sqrt of eigenvalues
        x_whitened = x_pca / np.sqrt(self.pca_explained_var + self.eps)

        # Step 4: ZCA rotation back to original basis
        zca_matrix = np.dot(self.pca_components.T / np.sqrt(self.pca_explained_var + self.eps), self.pca_components)
        x_zca = np.dot(x_centered, zca_matrix)

        return x_zca[0] if is_single else x_zca

    @classmethod
    def fit(cls, X: np.ndarray, pca_dim: int = 256, eps: float = 1e-8):
        """
        Fit PCA → ZCA whitening on embedding matrix X.
        """
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Step 1: PCA
        pca = PCA(n_components=pca_dim)
        pca.fit(X_centered)
        components = pca.components_
        explained_var = pca.explained_variance_

        return cls(mean, components, explained_var, eps)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'pca_components': self.pca_components,
                'pca_explained_var': self.pca_explained_var,
                'eps': self.eps
            }, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(
            mean=data['mean'],
            pca_components=data['pca_components'],
            pca_explained_var=data['pca_explained_var'],
            eps=data['eps']
        )

def encode_and_whiten_pcazca(sentences, st_model, whitening_model) -> np.ndarray:
    """
    Encode sentences with SentenceTransformer and whiten embeddings with PCA+ZCA whitening model.

    Args:
        sentences (list of str or np.ndarray): Sentences or precomputed embeddings.
        st_model (SentenceTransformer): SentenceTransformer model.
        whitening_model (PCAZCAWhiteningModel): Fitted whitening model.

    Returns:
        np.ndarray: Whitened embeddings.
    """
    if isinstance(sentences[0], str):
        embeddings = st_model.encode(sentences, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    else:
        embeddings = sentences

    whitened_embeddings = whitening_model.transform(embeddings)
    return whitened_embeddings
