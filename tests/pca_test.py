import numpy as np
from src.whitening.pca_ica import PCAICAWhiteningModel

# Assuming sts_embeddings and convo_embs are loaded from independent functions
def train_pcaica_whitening_model(sts_embeddings, convo_embs, emb_dim=512):
    """
    Trains the PCAICAWhiteningModel.

    Args:
        sts_embeddings (np.ndarray): Embeddings from the STS dataset.
        convo_embs (np.ndarray): Embeddings from the conversation corpus.
        emb_dim (int): The desired dimensionality after PCA.

    Returns:
        PCAICAWhiteningModel: The trained whitening model.
    """
    print(f"Training PCAICA Whitening Model with EMB_DIM={emb_dim}...")
    combined_embeddings = np.concatenate((sts_embeddings, convo_embs), axis=0)
    whitening_transform_model = PCAICAWhiteningModel.fit(combined_embeddings, emb_dim)
    print("PCAICA Whitening Model training complete.")
    return whitening_transform_model


# Assuming load_embeddings and load_sts_embeddings are defined and accessible
if __name__ == "__main__":
    
    from sentence_transformers import SentenceTransformer
    from src.utils.datasets import load_embeddings, load_sts_embeddings
    
    st_model = SentenceTransformer('all-roberta-large-v1', trust_remote_code=True)
    sts_embeddings, _ = load_sts_embeddings(st_model, split='train', score_threshold=0.0)
    if sts_embeddings.size > 0:
        EMB_DIM = 512
        whitening_model = train_pcaica_whitening_model(sts_embeddings, EMB_DIM)