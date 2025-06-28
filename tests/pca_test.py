import numpy as np
from src.whitening.pca_ica import PCAICAWhiteningModel

# Assuming sts_embeddings and convo_embs are loaded from independent functions
def train_pcaica_whitening_model(sts_embeddings, n_components=0.96):
    """
    Trains the PCAICAWhiteningModel.

    Args:
        sts_embeddings (np.ndarray): Embeddings from the STS dataset.
        convo_embs (np.ndarray): Embeddings from the conversation corpus.
        emb_dim (int): The desired dimensionality after PCA.

    Returns:
        PCAICAWhiteningModel: The trained whitening model.
    """
    if n_components < 0 or n_components > 1:
        print(f"Training PCAICA Whitening Model with variance ratio={n_components}...")
    else:
        print(f"Training PCAICA Whitening Model with EMB_DIM={n_components}...")
    whitening_transform_model = PCAICAWhiteningModel.fit(sts_embeddings, n_components)
    print("PCAICA Whitening Model training complete.")
    return whitening_transform_model


# Assuming load_embeddings and load_sts_embeddings are defined and accessible
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from src.utils.datasets import load_embeddings, load_sts_embeddings
    
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)
    sts_embeddings, _ = load_sts_embeddings(st_model, split='test', score_threshold=0.0)
    if sts_embeddings.size > 0:
        whitening_model = train_pcaica_whitening_model(sts_embeddings, n_components=0.96)
    print(whitening_model)