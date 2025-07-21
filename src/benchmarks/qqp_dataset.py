import os
import numpy as np
import time
import json
import hashlib
from tqdm import tqdm
from random import sample as randsample, shuffle
from tabulate import tabulate

from transformers import AutoTokenizer, T5EncoderModel
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import ndcg_score

import faiss
from annoy import AnnoyIndex
import hnswlib
from functools import partial

import pandas as pd

from src.cobweb.CobwebWrapper import CobwebWrapper
from src.whitening.pca_ica import PCAICAWhiteningModel as PCAICAWhitening


def get_embedding_path(model_name: str, dataset: str, split: str):
    os.makedirs("data/embeddings", exist_ok=True)
    model_name = model_name.replace('/', '-')
    return f"data/embeddings/{model_name}_{dataset}_{split}.npy"

def load_or_compute_embeddings(texts, model_name, dataset, split, compute = False):
    path = get_embedding_path(model_name, dataset, split)
    if os.path.exists(path) and not compute:
        print(f"Loading embeddings from {path}")
        return np.load(path)
    else:
        print(f"Computing embeddings and saving to {path}")
        print(f"texts : {texts[:5]}...")  # Show first 5 texts for debugging
        if "t5" in model_name:
            texts = ["Summarize :" + text for text in texts]
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = T5EncoderModel.from_pretrained(model_name, trust_remote_code=True)
            inputs = tokenizer(texts, return_tensors='pt', padding=True).input_ids
            with torch.no_grad():
                outputs = model(input_ids = inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        else:
            model = SentenceTransformer(model_name, trust_remote_code=True)
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        print(f"Computed embeddings shape: {embeddings.shape}")
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings for {model_name} should be 2D, got {embeddings.shape}. Check the model and input texts.")
        np.save(path, embeddings)
        return embeddings

def load_saved_sentences(targets, model_name, dataset, split, compute = False):
    path = f"data/sentences/{model_name.replace('/', '-')}_{dataset}_{split}.json"
    if targets is None:
        print(f"Loading sentences from {path}")
        with open(path, 'r') as f:
            return f.readlines()
    else:
        print(f"Saving sentences to {path}")
        with open(path, 'w+') as f:
            f.write('\n'.join(targets))
        return targets


def load_cobweb_model(model_name, corpus, corpus_embs, split, mode):
    cobweb_path = f"models/cobweb_wrappers/{model_name.replace('/', '-')}_{split}_{mode}.json"
    if os.path.exists(cobweb_path) and False:
        print(f"Loading Cobweb model from {cobweb_path}")
        with open(cobweb_path, 'r') as f:
            cobweb_json = json.load(f)
        return CobwebWrapper.load_json(cobweb_json, encode_func=lambda x: x)
    else:
        print(f"Computing Cobweb model and saving to {cobweb_path}")
        cobweb = CobwebWrapper(corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)
        cobweb.dump_json(cobweb_path)
        return cobweb

def load_pca_ica_model(corpus_embs, model_name, dataset, split):
    pca_dim=0.96
    pca_ica_path = f"models/pca_ica/{model_name.replace('/', '-')}_{dataset}_{split}_{'_'.join(str(pca_dim).split('.'))}.pkl"
    if os.path.exists(pca_ica_path):
        print(f"Loading PCA + ICA model from {pca_ica_path}")
        return PCAICAWhitening.load(pca_ica_path)
    else:
        print(f"Computing PCA + ICA model and saving to {pca_ica_path}")
        pca_ica_model = PCAICAWhitening.fit(corpus_embs, pca_dim=pca_dim)
        pca_ica_model.save(pca_ica_path)
        return pca_ica_model

def setup_cobweb_basic(corpus, corpus_embs):
    cobweb = CobwebWrapper(corpus = corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)
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

def setup_annoy(corpus_embs):
    dim = corpus_embs.shape[1]
    index = AnnoyIndex(dim, 'angular')
    for i, emb in enumerate(corpus_embs):
        index.add_item(i, emb)
    index.build(10)
    return index

def retrieve_annoy(query_emb, k, index, corpus):
    ids = index.get_nns_by_vector(query_emb, k)
    return [corpus[i] for i in ids]

def setup_hnsw(corpus_embs):
    dim = corpus_embs.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(corpus_embs), ef_construction=100, M=16)
    index.add_items(corpus_embs, np.arange(len(corpus_embs)))
    index.set_ef(50)
    return index

def retrieve_hnsw(query_emb, k, index, corpus):
    ids, _ = index.knn_query(query_emb, k=k)
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



# === Main Benchmark Runner ===

def run_qqp_benchmark(model_name, subset_size=7500, split = "test", target_size=750, top_k=3, compute = True):
    print(f"\n--- Running QQP Benchmark (TOP_K={top_k}) ---")

    corpus, queries, targets = None, None, None
    if compute:
        dataset = load_dataset("glue", "qqp", split=split, trust_remote_code=True)
        duplicates = [ex for ex in dataset if ex["label"] == 1]
        shuffle(duplicates)

        sampled = randsample(duplicates, subset_size)
        queries = [ex["question1"] for ex in sampled[:target_size]]
        targets = [ex["question2"] for ex in sampled[:target_size]]
        corpus = [ex["question2"] for ex in sampled]

        print("Corpus size:", len(corpus))

    corpus_embs = load_or_compute_embeddings(corpus, model_name, "qqp_corpus", split, compute = compute)
    print(f"Corpus embeddings shape: {corpus_embs.shape}")
    queries_embs = load_or_compute_embeddings(queries, model_name, "qqp_queries", split, compute = compute)
    print(f"Queries embeddings shape: {queries_embs.shape}")
    targets = load_saved_sentences(targets, model_name, "qqp_targets", split, compute = compute)
    corpus = load_saved_sentences(corpus, model_name, "qqp_corpus", split, compute = compute)

    pca_ica_model = load_pca_ica_model(corpus_embs, model_name, "qqp_corpus", split)
    print(f"PCA/ICA model loaded: {pca_ica_model}")

    print(f"Starting PCA and ICA embeddings transformation...")
    pca_corpus_embs = pca_ica_model.transform(corpus_embs, is_ica=False)
    pca_queries_embs = pca_ica_model.transform(queries_embs, is_ica=False)

    pca_ica_corpus_embs = pca_ica_model.transform(corpus_embs)
    pca_ica_queries_embs = pca_ica_model.transform(queries_embs)
    print(f"PCA and ICA embeddings transformation completed.")


    # Setup retrieval methods
    results = []
    save_path = f"outputs/qqp_benchmark_{model_name.replace('/', '-')}_c{subset_size}_t{target_size}_{split}.txt"
    print(f"Setting up FAISS...")
    faiss_index = setup_faiss(corpus_embs)
    results.append(evaluate_retrieval("FAISS", queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_index, corpus), top_k))
    print(f"--- FAISS Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    print(f"Setting up Basic Cobweb...")
    cobweb = load_cobweb_model(model_name, corpus, corpus_embs, split, "base")
    results.append(evaluate_retrieval("Cobweb", queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb), top_k))
    if len(results) < 2:
        print("ERROR")
    print(f"--- Basic Cobweb Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    results.append(evaluate_retrieval("Cobweb Fast", queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb, use_fast=True), top_k))
    print(f"--- Cobweb Fast Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)


    print(f"Setting up PCA Cobweb...")
    cobweb_pca = load_cobweb_model(model_name, corpus, pca_corpus_embs, split, "pca")
    results.append(evaluate_retrieval("Cobweb PCA", pca_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca), top_k))
    print(f"--- Cobweb PCA Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    print(f"Setting up PCA + ICA Cobweb...")
    cobweb_pca_ica = load_cobweb_model(model_name, corpus, pca_ica_corpus_embs, split, "pca_ica")
    results.append(evaluate_retrieval("Cobweb PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica), top_k))
    print(f"--- Cobweb PCA + ICA Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)
    results.append(evaluate_retrieval("Cobweb PCA + ICA Fast", pca_ica_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica, use_fast=True), top_k))
    print(f"--- Cobweb PCA + ICA Fast Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    return results

if __name__ == "__main__":
    model_name = 'all-roberta-large-v1'  # Example model
    # model_name = "google-t5/t5-base"
    results = run_qqp_benchmark(model_name, subset_size=1500, split="train", target_size=300, top_k=10, compute = True)  # Adjust split and top_k as needed
    for res in results:
        print_metrics_table(res)