"""
Utilities for benchmarking different retrieval methods.
Provides common functionality for embeddings, model loading, evaluation, and file naming.
"""

import os
import numpy as np
import time
import json
import hashlib
from tqdm import tqdm
from tabulate import tabulate
from typing import List, Dict, Any, Tuple, Optional, Callable

from transformers import (
    AutoTokenizer, T5EncoderModel,
    DPRQuestionEncoder, DPRContextEncoder, 
    DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
)
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

import faiss
from annoy import AnnoyIndex
import hnswlib

from src.cobweb.CobwebWrapper import CobwebWrapper
from src.whitening.pca_ica import PCAICAWhiteningModel as PCAICAWhitening
from src.whitening.pca_zca import PCAZCAWhiteningModel as PCAZCAWhitening
from src.whitening.zca import ZCAWhiteningModel as ZCAWhitening


def generate_unique_id(model_name: str, dataset: str, split: str, subset_size: int, 
                      target_size: int, **kwargs) -> str:
    """
    Generate a unique identifier based on all benchmark parameters.
    
    Args:
        model_name: Name of the model used
        dataset: Dataset name
        split: Data split (train/validation/test)
        subset_size: Size of the corpus subset
        target_size: Number of query-target pairs
        **kwargs: Additional parameters to include in the hash
    
    Returns:
        Unique identifier string
    """
    # Create a dictionary with all parameters
    params = {
        'model_name': model_name.replace('/', '-'),
        'dataset': dataset,
        'split': split,
        'subset_size': subset_size,
        'target_size': target_size,
        **kwargs
    }
    
    # Create a sorted string representation
    param_str = '_'.join([f"{k}={v}" for k, v in sorted(params.items())])
    
    # Generate a hash for very long parameter strings
    if len(param_str) > 100:
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        base_params = f"{params['model_name']}_{params['dataset']}_{params['split']}_c{params['subset_size']}_t{params['target_size']}_k{params['top_k']}"
        return f"{base_params}_{param_hash}"
    
    return param_str.replace('=', '').replace('/', '-')


def get_embedding_path(model_name: str, dataset: str, split: str, 
                      model_type: Optional[str] = None, unique_id: Optional[str] = None) -> str:
    """
    Generate path for embedding files with optional unique identifier.
    
    Args:
        model_name: Name of the model
        dataset: Dataset name
        split: Data split
        model_type: Optional model type (e.g., 'query', 'passage')
        unique_id: Optional unique identifier for the run
    
    Returns:
        Path to embedding file
    """
    os.makedirs("data/embeddings", exist_ok=True)
    model_name = model_name.replace('/', '-')
    
    if model_type and unique_id:
        return f"data/embeddings/{model_type}_{model_name}_{dataset}_{split}_{unique_id}.npy"
    elif model_type:
        return f"data/embeddings/{model_type}_{model_name}_{dataset}_{split}.npy"
    elif unique_id:
        return f"data/embeddings/{model_name}_{dataset}_{split}_{unique_id}.npy"
    else:
        return f"data/embeddings/{model_name}_{dataset}_{split}.npy"


def get_sentences_path(model_name: str, dataset: str, split: str, unique_id: Optional[str] = None) -> str:
    """Generate path for saved sentences with optional unique identifier."""
    os.makedirs("data/sentences", exist_ok=True)
    model_name = model_name.replace('/', '-')
    
    if unique_id:
        return f"data/sentences/{model_name}_{dataset}_{split}_{unique_id}.json"
    else:
        return f"data/sentences/{model_name}_{dataset}_{split}.json"


def get_model_path(model_name: str, split: str, mode: str, model_type: str, unique_id: Optional[str] = None) -> str:
    """
    Generate path for saved models with optional unique identifier.
    
    Args:
        model_name: Name of the model
        split: Data split
        mode: Model mode (e.g., 'base', 'pca_ica', 'zca')
        model_type: Type of model (e.g., 'cobweb_wrappers', 'pca_ica', 'zca')
        unique_id: Optional unique identifier
    
    Returns:
        Path to model file
    """
    os.makedirs(f"models/{model_type}", exist_ok=True)
    model_name = model_name.replace('/', '-')
    
    if unique_id:
        if model_type == 'cobweb_wrappers':
            return f"models/{model_type}/{model_name}_{split}_{mode}_{unique_id}.json"
        else:
            return f"models/{model_type}/{model_name}_{split}_{mode}_{unique_id}.pkl"
    else:
        if model_type == 'cobweb_wrappers':
            return f"models/{model_type}/{model_name}_{split}_{mode}.json"
        else:
            return f"models/{model_type}/{model_name}_{split}_{mode}.pkl"


def get_results_path(model_name: str, dataset: str, split: str, unique_id: str) -> str:
    """Generate path for benchmark results with unique identifier."""
    os.makedirs(f"outputs/{dataset}", exist_ok=True)
    model_name = model_name.replace('/', '-')
    return f"outputs/{dataset}/benchmark_{model_name}_{split}_{unique_id}.txt"


def load_or_compute_embeddings(texts: List[str], model_name: str, dataset: str, split: str, 
                             compute: bool = False, unique_id: Optional[str] = None) -> np.ndarray:
    """
    Load or compute embeddings using SentenceTransformer or T5.
    
    Args:
        texts: List of texts to encode
        model_name: Name of the model to use
        dataset: Dataset name
        split: Data split
        compute: Whether to force recomputation
        unique_id: Optional unique identifier for this run
    
    Returns:
        Numpy array of embeddings
    """
    path = get_embedding_path(model_name, dataset, split, unique_id=unique_id)
    
    if os.path.exists(path) and not compute:
        print(f"Loading embeddings from {path}")
        return np.load(path)
    else:
        print(f"Computing embeddings and saving to {path}")
        print(f"texts : {texts[:5]}...")  # Show first 5 texts for debugging
        
        if "t5" in model_name:
            # Add task prefix for T5
            texts = ["Summarize: " + text for text in texts]
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = T5EncoderModel.from_pretrained(model_name, trust_remote_code=True)
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids
            with torch.no_grad():
                outputs = model(input_ids=inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        else:
            model = SentenceTransformer(model_name, trust_remote_code=True)
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        print(f"Computed embeddings shape: {embeddings.shape}")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings for {model_name} should be 2D, got {embeddings.shape}. Check the model and input texts.")
        
        np.save(path, embeddings)
        return embeddings


def load_or_compute_dpr_embeddings(texts: List[str], model_type: str, model_name: str, 
                                 dataset: str, split: str, compute: bool = False, 
                                 unique_id: Optional[str] = None) -> np.ndarray:
    """
    Load or compute DPR embeddings for either queries or passages.
    
    Args:
        texts: List of texts to encode
        model_type: Either 'query' or 'passage'
        model_name: Base DPR model name
        dataset: Dataset name
        split: Data split
        compute: Whether to force recomputation
        unique_id: Optional unique identifier for this run
    
    Returns:
        Numpy array of embeddings
    """
    path = get_embedding_path(model_name, dataset, split, model_type=model_type, unique_id=unique_id)
    
    if os.path.exists(path) and not compute:
        print(f"Loading {model_type} embeddings from {path}")
        return np.load(path)
    else:
        print(f"Computing {model_type} embeddings and saving to {path}")
        print(f"texts : {texts[:5]}...")  # Show first 5 texts for debugging
        
        if model_type == 'query':
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
            model = DPRQuestionEncoder.from_pretrained(model_name)
        elif model_type == 'passage':
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
            model = DPRContextEncoder.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Process in batches to avoid memory issues
        batch_size = 10
        embeddings = []
        model.eval()
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing {model_type} embeddings"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.pooler_output.numpy()
                embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"Computed {model_type} embeddings shape: {embeddings.shape}")
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings for {model_type} should be 2D, got {embeddings.shape}. Check the model and input texts.")
        
        np.save(path, embeddings)
        return embeddings


def load_or_save_sentences(sentences: Optional[List[str]], model_name: str, dataset: str, 
                          split: str, compute: bool = False, unique_id: Optional[str] = None) -> List[str]:
    """
    Load or save sentences to/from file.
    
    Args:
        sentences: List of sentences to save (None to load)
        model_name: Name of the model
        dataset: Dataset name
        split: Data split
        compute: Whether to force saving
        unique_id: Optional unique identifier for this run
    
    Returns:
        List of sentences
    """
    path = get_sentences_path(model_name, dataset, split, unique_id=unique_id)
    
    if sentences is None:
        print(f"Loading sentences from {path}")
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        print(f"Saving sentences to {path}")
        with open(path, 'w+') as f:
            f.write('\n'.join(sentences))
        return sentences


def load_cobweb_model(model_name: str, corpus: List[str], corpus_embs: np.ndarray, 
                     split: str, mode: str, unique_id: Optional[str] = None, 
                     force_compute: bool = False) -> CobwebWrapper:
    """
    Load or create a Cobweb model.
    
    Args:
        model_name: Name of the model
        corpus: List of corpus texts
        corpus_embs: Corpus embeddings
        split: Data split
        mode: Model mode (e.g., 'base', 'pca_ica')
        unique_id: Optional unique identifier
        force_compute: Whether to force recomputation
    
    Returns:
        CobwebWrapper instance
    """
    cobweb_path = get_model_path(model_name, split, mode, 'cobweb_wrappers', unique_id)
    
    if os.path.exists(cobweb_path) and not force_compute:
        print(f"Loading Cobweb model from {cobweb_path}")
        with open(cobweb_path, 'r') as f:
            cobweb_json = json.load(f)
        return CobwebWrapper.load_json(cobweb_json, encode_func=lambda x: x)
    else:
        print(f"Computing Cobweb model and saving to {cobweb_path}")
        cobweb = CobwebWrapper(corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)
        cobweb.dump_json(cobweb_path)
        return cobweb


def load_pca_ica_model(corpus_embs: np.ndarray, model_name: str, dataset: str, split: str, 
                      model_type: str, target_dim: float = 0.96, unique_id: Optional[str] = None) -> PCAICAWhitening:
    """
    Load or create a PCA + ICA whitening model.
    
    Args:
        corpus_embs: Corpus embeddings to fit the model
        model_name: Name of the base model
        dataset: Dataset name
        split: Data split
        model_type: Model type identifier
        target_dim: Target dimensionality (as fraction or int)
        unique_id: Optional unique identifier
    
    Returns:
        PCAICAWhitening model
    """
    dim_str = '_'.join(str(target_dim).split('.'))
    mode = f"{model_type}_{dim_str}"
    pca_ica_path = get_model_path(model_name, split, mode, 'pca_ica', unique_id)
    
    if os.path.exists(pca_ica_path):
        print(f"Loading PCA + ICA model for {model_type} from {pca_ica_path}")
        return PCAICAWhitening.load(pca_ica_path)
    else:
        print(f"Computing PCA + ICA model for {model_type} and saving to {pca_ica_path}")
        pca_ica_model = PCAICAWhitening.fit(corpus_embs, pca_dim=target_dim)
        pca_ica_model.save(pca_ica_path)
        return pca_ica_model


def load_zca_model(corpus_embs: np.ndarray, model_name: str, dataset: str, split: str, 
                  model_type: str, unique_id: Optional[str] = None) -> ZCAWhitening:
    """
    Load or create a ZCA whitening model.
    
    Args:
        corpus_embs: Corpus embeddings to fit the model
        model_name: Name of the base model
        dataset: Dataset name
        split: Data split
        model_type: Model type identifier
        unique_id: Optional unique identifier
    
    Returns:
        ZCAWhitening model
    """
    zca_path = get_model_path(model_name, split, model_type, 'zca', unique_id)
    
    if os.path.exists(zca_path):
        print(f"Loading ZCA model for {model_type} from {zca_path}")
        return ZCAWhitening.load(zca_path)
    else:
        print(f"Computing ZCA model for {model_type} and saving to {zca_path}")
        zca_model = ZCAWhitening.fit(corpus_embs)
        zca_model.save(zca_path)
        return zca_model


# === Retrieval Setup Functions ===

def setup_cobweb_basic(corpus: List[str], corpus_embs: np.ndarray) -> CobwebWrapper:
    """Setup a basic Cobweb wrapper."""
    return CobwebWrapper(corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)


def setup_faiss(corpus_embs: np.ndarray) -> faiss.IndexFlatIP:
    """Setup a FAISS index for inner product similarity."""
    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs)
    return index


def setup_annoy(corpus_embs: np.ndarray) -> AnnoyIndex:
    """Setup an Annoy index for approximate nearest neighbor search."""
    dim = corpus_embs.shape[1]
    index = AnnoyIndex(dim, 'angular')
    for i, emb in enumerate(corpus_embs):
        index.add_item(i, emb)
    index.build(10)
    return index


def setup_hnsw(corpus_embs: np.ndarray) -> hnswlib.Index:
    """Setup an HNSW index for approximate nearest neighbor search."""
    dim = corpus_embs.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(corpus_embs), ef_construction=100, M=16)
    index.add_items(corpus_embs, np.arange(len(corpus_embs)))
    index.set_ef(50)
    return index


# === Retrieval Functions ===

def retrieve_cobweb_basic(query_emb: np.ndarray, k: int, cobweb: CobwebWrapper, use_fast: bool = False) -> List[str]:
    """Retrieve using Cobweb wrapper."""
    if use_fast:
        return cobweb.cobweb_predict_fast(query_emb, k)
    else:
        return cobweb.cobweb_predict(query_emb, k)


def retrieve_faiss(query_emb: np.ndarray, k: int, index: faiss.IndexFlatIP, corpus: List[str]) -> List[str]:
    """Retrieve using FAISS index."""
    _, ids = index.search(np.expand_dims(query_emb, axis=0), k)
    return [corpus[i] for i in ids[0]]


def retrieve_annoy(query_emb: np.ndarray, k: int, index: AnnoyIndex, corpus: List[str]) -> List[str]:
    """Retrieve using Annoy index."""
    ids = index.get_nns_by_vector(query_emb, k)
    return [corpus[i] for i in ids]


def retrieve_hnsw(query_emb: np.ndarray, k: int, index: hnswlib.Index, corpus: List[str]) -> List[str]:
    """Retrieve using HNSW index."""
    ids, _ = index.knn_query(query_emb, k=k)
    return [corpus[i] for i in ids[0]]


# === Evaluation Functions ===

def get_eval_ks(top_k: int) -> List[int]:
    """Return a list of k-values to evaluate based on top_k."""
    base = [2, 3, 5, 10, 20, 50, 100]
    return sorted([k for k in base if k <= top_k])


def evaluate_retrieval(name: str, queries: List[np.ndarray], targets: List[str], 
                      retrieve_fn: Callable, top_k: int = 10) -> Dict[str, Any]:
    """
    Evaluate a retrieval method using multiple metrics.
    
    Args:
        name: Name of the retrieval method
        queries: List of query embeddings
        targets: List of target strings (ground truth)
        retrieve_fn: Function to retrieve results given (query, k)
        top_k: Maximum number of results to retrieve
    
    Returns:
        Dictionary containing evaluation metrics
    """
    ks = get_eval_ks(top_k)
    metrics = {
        f"recall@{k}": 0 for k in ks
    }
    metrics.update({
        f"mrr@{k}": 0 for k in ks
    })
    metrics.update({
        f"ndcg@{k}": 0 for k in ks
    })

    latencies = []

    for query, target in tqdm(zip(queries, targets), total=len(queries), desc=f"Evaluating {name}"):
        start = time.time()
        retrieved = retrieve_fn(query, top_k)
        latencies.append(time.time() - start)

        for k in ks:
            top_k_results = retrieved[:k]
            if target in top_k_results:
                metrics[f"recall@{k}"] += 1
                rank = top_k_results.index(target) + 1
                metrics[f"mrr@{k}"] += 1 / rank
            relevance = [1 if doc == target else 0 for doc in top_k_results]
            if sum(relevance) > 0:
                ideal = sorted(relevance, reverse=True)
                ndcg = ndcg_score([ideal], [relevance])
                metrics[f"ndcg@{k}"] += ndcg

    n = len(queries)
    for k in ks:
        metrics[f"recall@{k}"] = round(metrics[f"recall@{k}"] / n, 4)
        metrics[f"mrr@{k}"] = round(metrics[f"mrr@{k}"] / n, 4)
        metrics[f"ndcg@{k}"] = round(metrics[f"ndcg@{k}"] / n, 4)
    
    metrics["time_taken"] = round(np.sum(latencies), 2)
    metrics["method"] = name
    metrics["avg_latency_ms"] = round(1000 * np.mean(latencies), 2)

    return metrics


def print_metrics_table(metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Print and optionally save metrics in a formatted table.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_path: Optional path to save the results
    """
    method = metrics.pop("method", "Unknown")
    latency = metrics.pop("avg_latency_ms", None)
    total_time = metrics.pop("time_taken", 0)

    rows = []
    ks = sorted(set(int(k.split('@')[1]) for k in metrics if '@' in k))
    for k in ks:
        row = [
            f"@{k}",
            metrics.get(f"recall@{k}", 0),
            metrics.get(f"mrr@{k}", 0),
            metrics.get(f"ndcg@{k}", 0)
        ]
        rows.append(row)

    table_str = f"\n--- Metrics for {method} ---\n"
    if latency is not None:
        table_str += f"Avg Latency: {latency} ms with total time {total_time} seconds\n"
    table_str += tabulate(rows, headers=["k", "Recall", "MRR", "nDCG"], tablefmt="pretty")

    print(table_str)

    if save_path:
        with open(save_path, "a+") as f: 
            f.write(table_str + "\n")
