from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

DATA_DIR = "data"

def download_dataset(name:str, save_path:str):
    """
    Downloads a dataset from Hugging Face and saves it to the specified path.
    
    Args:
        name (str): The name of the dataset to download.
        save_path (str): The path where the dataset will be saved.
    """
    dataset = load_dataset(name)
    dataset.save_to_disk(save_path)
    print(f"Dataset {name} downloaded and saved to {save_path}.")


def dataset_to_embeddings(dataset, model_name:str):
    """
    Converts a dataset to embeddings using a specified model.
    
    Args:
        dataset: The dataset to convert.
        model_name (str): The name of the model to use for embedding.
        
    Returns:
        list: A list of embeddings.
    """
    # if dataset == 'stsb_multi_mt':
    #     if 

    model = SentenceTransformer(model_name)
    embeddings = model.encode(dataset['text'], show_progress_bar=True)
    return embeddings



def load_sts_dataset(model_name='all-roberta-large-v1', split='train', path = DATA_DIR +"/embeddings", score_threshold=None):
    # Load STS dataset
    dataset = load_dataset("stsb_multi_mt", name="en", split=split)

    # Load sentence transformer model
    st_model = SentenceTransformer(model_name)

    embeddings = []
    labels = []

    for item in tqdm(dataset, desc = f"Processing STS {split} split..."):
        s1 = item['sentence1']
        s2 = item['sentence2']
        score = item['similarity_score'] / 5.0  # Normalize to [0, 1]

        # Optional: Only use highly similar pairs (e.g., for VAE focusing on fine semantics)
        if score_threshold is not None and score < score_threshold:
            continue

        # Get embeddings
        emb1 = st_model.encode(s1, convert_to_numpy=True)
        emb2 = st_model.encode(s2, convert_to_numpy=True)

        # Option 1: Use both individually
        embeddings.append(emb1)
        embeddings.append(emb2)

        # Option 2 (alt): Use difference, mean, or concat if you prefer contrastive-style input
        # embeddings.append(np.abs(emb1 - emb2))

        labels.append(score)
        labels.append(score)  # Both emb1 and emb2 share the score (if used individually)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels
