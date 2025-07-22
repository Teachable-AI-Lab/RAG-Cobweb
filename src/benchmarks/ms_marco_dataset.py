import os
import numpy as np
import time
import json
import hashlib
from tqdm import tqdm
from random import sample as randsample, shuffle
from tabulate import tabulate

from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch
from datasets import load_dataset
from sklearn.metrics import ndcg_score

import faiss
from annoy import AnnoyIndex
import hnswlib
from functools import partial

from src.cobweb.CobwebWrapper import CobwebWrapper
from src.whitening.pca_ica import PCAICAWhiteningModel as PCAICAWhitening
from src.whitening.zca import ZCAWhiteningModel as ZCAWhitening

def get_embedding_path(model_type: str, model_name: str, dataset: str, split: str):
    os.makedirs("data/embeddings", exist_ok=True)
    model_name = model_name.replace('/', '-')
    return f"data/embeddings/{model_type}_{model_name}_{dataset}_{split}.npy"

def load_or_compute_dpr_embeddings(texts, model_type, model_name, dataset, split, compute=False):
    """
    Load or compute DPR embeddings for either queries or passages.
    
    Args:
        texts: List of texts to encode
        model_type: Either 'query' or 'passage'
        model_name: Base DPR model name
        dataset: Dataset name
        split: Data split
        compute: Whether to force recomputation
    """
    path = get_embedding_path(model_type, model_name, dataset, split)
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

def load_saved_sentences(targets, model_name, dataset, split, compute=False):
    path = f"data/sentences/{model_name.replace('/', '-')}_{dataset}_{split}.json"
    os.makedirs("data/sentences", exist_ok=True)
    if targets is None:
        print(f"Loading sentences from {path}")
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        print(f"Saving sentences to {path}")
        with open(path, 'w+') as f:
            f.write('\n'.join(targets))
        return targets

def load_cobweb_model(model_name, corpus, corpus_embs, split, mode):
    os.makedirs("models/cobweb_wrappers", exist_ok=True)
    cobweb_path = f"models/cobweb_wrappers/{model_name.replace('/', '-')}_{split}_{mode}.json"
    if os.path.exists(cobweb_path) and False:  # Set to True to enable loading
        print(f"Loading Cobweb model from {cobweb_path}")
        with open(cobweb_path, 'r') as f:
            cobweb_json = json.load(f)
        return CobwebWrapper.load_json(cobweb_json, encode_func=lambda x: x)
    else:
        print(f"Computing Cobweb model and saving to {cobweb_path}")
        cobweb = CobwebWrapper(corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)
        cobweb.dump_json(cobweb_path)
        return cobweb

def load_pca_ica_model(corpus_embs, model_name, dataset, split, model_type, target_dim=0.96):
    os.makedirs("models/pca_ica", exist_ok=True)
    pca_ica_path = f"models/pca_ica/{model_type}_{model_name.replace('/', '-')}_{dataset}_{split}_{'_'.join(str(target_dim).split('.'))}.pkl"
    if os.path.exists(pca_ica_path):
        print(f"Loading PCA + ICA model for {model_type} from {pca_ica_path}")
        return PCAICAWhitening.load(pca_ica_path)
    else:
        print(f"Computing PCA + ICA model for {model_type} and saving to {pca_ica_path}")
        pca_ica_model = PCAICAWhitening.fit(corpus_embs, pca_dim=target_dim)
        pca_ica_model.save(pca_ica_path)
        return pca_ica_model

def load_zca_model(corpus_embs, model_name, dataset, split, model_type):
    os.makedirs("models/zca", exist_ok=True)
    zca_path = f"models/zca/{model_type}_{model_name.replace('/', '-')}_{dataset}_{split}.pkl"
    if os.path.exists(zca_path):
        print(f"Loading ZCA model for {model_type} from {zca_path}")
        return ZCAWhitening.load(zca_path)
    else:
        print(f"Computing ZCA model for {model_type} and saving to {zca_path}")
        zca_model = ZCAWhitening.fit(corpus_embs)
        zca_model.save(zca_path)
        return zca_model

def setup_cobweb_basic(corpus, corpus_embs):
    cobweb = CobwebWrapper(corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)
    return cobweb

def retrieve_cobweb_basic(query_emb, k, cobweb, use_fast=False):
    if use_fast:
        return cobweb.cobweb_predict_fast(query_emb, k)
    else:
        return cobweb.cobweb_predict(query_emb, k)
    
def setup_faiss(corpus_embs):
    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs)
    return index

def retrieve_faiss(query_emb, k, index, corpus):
    _, ids = index.search(np.expand_dims(query_emb, axis=0), k)
    return [corpus[i] for i in ids[0]]

# === Evaluation Function ===

def get_eval_ks(top_k):
    """Return a list of k-values to evaluate based on top_k."""
    base = [3, 5, 10, 20, 50, 100]
    return sorted([k for k in base if k <= top_k])

def evaluate_retrieval(name, queries, targets, retrieve_fn, top_k=10):
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

def print_metrics_table(metrics, save_path=None):
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

# === Main MS Marco Benchmark Runner ===

def run_msmarco_benchmark(dpr_model_name="facebook/dpr-question_encoder-single-nq-base", subset_size=7500, split="validation", target_size=750, top_k=3, compute=True):
    print(f"\n--- Running MS Marco Benchmark with DPR (TOP_K={top_k}) ---")
    
    # DPR model names
    query_model_name = dpr_model_name
    passage_model_name = dpr_model_name.replace("question_encoder", "ctx_encoder")
    
    print(f"Using Query Model: {query_model_name}")
    print(f"Using Passage Model: {passage_model_name}")

    corpus, queries, targets = None, None, None
    if compute:
        # Load MS Marco dataset
        print("Loading MS Marco dataset...")
        ds = load_dataset("ms_marco", "v2.1", split=split)
        ds.shuffle()
        all_passages = []
        positive_pairs = []
        all_passages = []
        corpus = []
        for ex in ds:
            query = ex["query"]
            passage_texts = ex["passages"]["passage_text"]
            is_selected_flags = ex["passages"]["is_selected"]
            if any(is_selected_flags) and len(positive_pairs) < target_size:
                positive_pairs.append((query, passage_texts[is_selected_flags.index(1)]))
                corpus.extend(passage_texts)
            elif len(corpus) < subset_size:
                all_passages.extend(passage_texts)
            else:
                break
        if len(corpus) < subset_size:
            corpus.extend(randsample(all_passages, subset_size - len(corpus)))
        queries = [pair[0] for pair in positive_pairs]
        targets = [pair[1] for pair in positive_pairs]
        print(f"Loaded {len(corpus)} passages, {len(queries)} queries, and {len(targets)} targets.")
    
        
                

    # Load or compute embeddings using separate DPR models
    corpus_embs = load_or_compute_dpr_embeddings(corpus, "passage", passage_model_name, "msmarco_corpus", split, compute=compute)
    print(f"Corpus embeddings shape: {corpus_embs.shape}")
    
    queries_embs = load_or_compute_dpr_embeddings(queries, "query", query_model_name, "msmarco_queries", split, compute=compute)
    print(f"Queries embeddings shape: {queries_embs.shape}")
    
    # Load saved sentences
    targets = load_saved_sentences(targets, query_model_name, "msmarco_targets", split, compute=compute)
    corpus = load_saved_sentences(corpus, passage_model_name, "msmarco_corpus", split, compute=compute)

    pca_ica_model = load_pca_ica_model(np.vstack((corpus_embs, queries_embs)), passage_model_name, "msmarco_general", split, "passage", target_dim=0.96)
    print(f"PCA + ICA model loaded: {pca_ica_model}")
    pca_ica_corpus_embs = pca_ica_model.transform(corpus_embs)
    pca_ica_queries_embs = pca_ica_model.transform(queries_embs)

    # Load separate PCA/ICA models for queries and passages
    # passage_pca_ica_model = load_pca_ica_model(corpus_embs, passage_model_name, "msmarco_corpus", split, "passage", target_dim=0.96)
    # pca_ica_corpus_embs = passage_pca_ica_model.transform(corpus_embs)
    # query_target_dim = pca_ica_corpus_embs.shape[1]  # Use the same target dimension as corpus for queries
    # query_pca_ica_model = load_pca_ica_model(queries_embs, query_model_name, "msmarco_queries", split, "query", target_dim=query_target_dim)
    # print(f"Query PCA/ICA model loaded: {query_pca_ica_model}")
    # print(f"Passage PCA/ICA model loaded: {passage_pca_ica_model}")

    # print(f"Starting PCA and ICA embeddings transformation...")

    # pca_ica_queries_embs = query_pca_ica_model.transform(queries_embs)
    # print(f"PCA and ICA embeddings transformation completed.")

    # Load separate ZCA models for queries and passages
    # passage_zca_model = load_zca_model(corpus_embs, passage_model_name, "msmarco_corpus", split, "passage", target_dim=0.96)



    # query_zca_model = load_zca_model(queries_embs, query_model_name, "msmarco_queries", split, "query", )


    
    # print(f"Query ZCA model loaded: {query_zca_model}")
    # print(f"Passage ZCA model loaded: {passage_zca_model}")

    # # print(f"Starting ZCA embeddings transformation...")
    # # zca_corpus_embs = passage_zca_model.transform(corpus_embs)
    # # zca_queries_embs = query_zca_model.transform(queries_embs)
    # # print(f"ZCA embeddings transformation completed.")

    # Setup retrieval methods
    results = []
    save_path = f"outputs/ms_marco/benchmark_{query_model_name.replace('/', '-')}_c{subset_size}_t{target_size}_{split}_dpr.txt"
    
    print(f"Setting up FAISS...")
    faiss_index = setup_faiss(corpus_embs)
    results.append(evaluate_retrieval("FAISS", queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_index, corpus), top_k))
    print(f"--- FAISS Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    print(f"Setting up Basic Cobweb...")
    cobweb = load_cobweb_model(query_model_name, corpus, corpus_embs, split, "base")
    results.append(evaluate_retrieval("Cobweb Basic", queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb), top_k))
    print(f"--- Cobweb Basic Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    print(f"Setting up PCA + ICA Cobweb...")
    cobweb_pca_ica = load_cobweb_model(query_model_name, corpus, pca_ica_corpus_embs, split, "pca_ica")
    results.append(evaluate_retrieval("Cobweb PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica), top_k))
    print(f"--- Cobweb PCA + ICA Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    # print(f"Setting up ZCA Cobweb...")
    # cobweb_zca = load_cobweb_model(query_model_name, corpus, zca_corpus_embs, split, "zca")
    # results.append(evaluate_retrieval("Cobweb ZCA", zca_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_zca), top_k))
    # print(f"--- Cobweb ZCA Metrics ---")
    # print_metrics_table(results[-1], save_path=save_path)

    return results

if __name__ == "__main__":
    dpr_model_name = 'facebook/dpr-question_encoder-single-nq-base'  # DPR model
    results = run_msmarco_benchmark(dpr_model_name, split="validation", top_k=10, compute=False)  # Adjust split and top_k as needed
    for res in results:
        print_metrics_table(res)