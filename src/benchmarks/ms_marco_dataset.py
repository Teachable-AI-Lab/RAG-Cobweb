import os
import argparse
import json
import numpy as np
from random import sample as randsample
from datasets import load_dataset

from src.utils.benchmark_utils import (
    generate_unique_id, load_or_compute_embeddings, load_or_compute_dpr_embeddings,
    load_or_save_sentences, load_cobweb_model, load_pca_ica_model,
    setup_faiss, retrieve_faiss, retrieve_cobweb_basic,
    evaluate_retrieval, print_metrics_table, get_results_path, setup_hnswlib, retrieve_hnswlib
)

def load_msmarco_dataset(subset_size=7500, split="validation", target_size=750):
    # Load MS Marco dataset and extract query-target pairs
    ds = load_dataset("ms_marco", "v2.1", split=split)
    ds.shuffle()
    
    all_passages = []
    positive_pairs = []
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
    
    return corpus, queries, targets

def get_benchmark_list(method = "all"):
    if method == "all":
        return ['FAISS', 'FAISS PCA + ICA', 'Cobweb Basic', 'Cobweb PCA + ICA']
    elif method == "extra":
        return ['FAISS', 'FAISS PCA + ICA', 'FAISS L2', 'FAISS L2 PCA + ICA', 'HNSWLib', 'HNSWLib PCA + ICA', 'Cobweb Basic', 'Cobweb PCA + ICA']
    elif method == "cobweb":
        return ['Cobweb Basic', 'Cobweb PCA + ICA']
    else:
        raise ValueError(f"Unknown method: {method}")

def run_msmarco_benchmark(model_name="facebook/dpr-question_encoder-single-nq-base", subset_size=7500, split="validation", target_size=750, top_k=3, compute=True, method="all"):
    print(f"\n--- Running MS Marco Benchmark (TOP_K={top_k}) ---")
    
    # Check if this is a DPR model or a general embedding model
    is_dpr_model = "dpr-" in model_name and ("question_encoder" in model_name or "ctx_encoder" in model_name)
    
    if is_dpr_model:
        print(f"Using DPR model: {model_name}")
        # DPR model names
        query_model_name = model_name
        passage_model_name = model_name.replace("question_encoder", "ctx_encoder")
        print(f"Using Query Model: {query_model_name}")
        print(f"Using Passage Model: {passage_model_name}")
    else:
        print(f"Using same model for both queries and passages: {model_name}")
        query_model_name = model_name
        passage_model_name = model_name

    # Generate unique identifier for this run
    unique_id = generate_unique_id(
        model_name=model_name,
        dataset="msmarco",
        split=split,
        top_k=top_k,
        subset_size=subset_size,
        target_size=target_size,
        is_dpr=is_dpr_model
    )
    print(f"Run ID: {unique_id}")

    corpus, queries, targets = None, None, None
    if compute:
        # Load MS Marco dataset
        print("Loading MS Marco dataset...")
        corpus, queries, targets = load_msmarco_dataset(subset_size, split, target_size)
        print(f"Loaded {len(corpus)} corpus entries, {len(queries)} queries, and {len(targets)} targets.")

    # Load or compute embeddings
    if is_dpr_model:
        # Use separate DPR models for queries and passages
        corpus_embs = load_or_compute_dpr_embeddings(corpus, "passage", passage_model_name, "msmarco_corpus", split, compute=compute, unique_id=unique_id)
        print(f"Corpus embeddings shape: {corpus_embs.shape}")
        
        queries_embs = load_or_compute_dpr_embeddings(queries, "query", query_model_name, "msmarco_queries", split, compute=compute, unique_id=unique_id)
        print(f"Queries embeddings shape: {queries_embs.shape}")
    else:
        # Use the same model for both queries and passages
        corpus_embs = load_or_compute_embeddings(corpus, model_name, "msmarco_corpus", split, compute=compute, unique_id=unique_id)
        print(f"Corpus embeddings shape: {corpus_embs.shape}")
        
        queries_embs = load_or_compute_embeddings(queries, model_name, "msmarco_queries", split, compute=compute, unique_id=unique_id)
        print(f"Queries embeddings shape: {queries_embs.shape}")
    
    # Load saved sentences
    targets = load_or_save_sentences(targets, query_model_name, "msmarco_targets", split, compute=compute, unique_id=unique_id)
    corpus = load_or_save_sentences(corpus, passage_model_name, "msmarco_corpus", split, compute=compute, unique_id=unique_id)

    pca_ica_model = load_pca_ica_model(np.vstack((corpus_embs, queries_embs)), passage_model_name, "msmarco", split, "general", target_dim=0.96, unique_id=unique_id)
    print(f"PCA + ICA model loaded: {pca_ica_model}")
    pca_ica_corpus_embs = pca_ica_model.transform(corpus_embs)
    pca_ica_queries_embs = pca_ica_model.transform(queries_embs)

    # Setup retrieval methods
    results = []
    save_path = get_results_path(query_model_name, "ms_marco", split, unique_id)
    
    get_benchmarks = get_benchmark_list(method)
    if 'FAISS' in get_benchmarks:
        print(f"Setting up FAISS...")
        faiss_index = setup_faiss(corpus_embs)
        results.append(evaluate_retrieval("FAISS", queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_index, corpus), top_k))
        print(f"--- FAISS Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)
    if 'FAISS PCA + ICA' in get_benchmarks:
        print(f"Setting up FAISS with PCA + ICA...")
        faiss_pca_ica_index = setup_faiss(pca_ica_corpus_embs)
        results.append(evaluate_retrieval("FAISS PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_pca_ica_index, corpus), top_k))
        print(f"--- FAISS PCA + ICA Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)
    if 'FAISS L2' in get_benchmarks:
        print(f"Setting up FAISS with L2 distance...")
        faiss_l2_index = setup_faiss(corpus_embs, index_type='l2')
        results.append(evaluate_retrieval("FAISS L2", queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_l2_index, corpus), top_k))
        print(f"--- FAISS L2 Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)
    if 'FAISS L2 PCA + ICA' in get_benchmarks:
        print(f"Setting up FAISS with L2 distance and PCA + ICA...")
        faiss_l2_pca_ica_index = setup_faiss(pca_ica_corpus_embs, index_type='l2')
        results.append(evaluate_retrieval("FAISS L2 PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_l2_pca_ica_index, corpus), top_k))
        print(f"--- FAISS L2 PCA + ICA Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)
    if 'HNSWLib' in get_benchmarks:
        print(f"Setting up HNSWLib...")
        hnswlib_index = setup_hnswlib(corpus_embs)
        results.append(evaluate_retrieval("HNSWLib", queries_embs, targets, lambda q, k: retrieve_hnswlib(q, k, hnswlib_index, corpus), top_k))
        print(f"--- HNSWLib Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)
    if 'HNSWLib PCA + ICA' in get_benchmarks:
        print(f"Setting up HNSWLib with PCA + ICA...")
        hnswlib_pca_ica_index = setup_hnswlib(pca_ica_corpus_embs)
        results.append(evaluate_retrieval("HNSWLib PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_hnswlib(q, k, hnswlib_pca_ica_index, corpus), top_k))
        print(f"--- HNSWLib PCA + ICA Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)
    if 'Cobweb Basic' in get_benchmarks:
        print(f"Setting up Basic Cobweb...")
        cobweb = load_cobweb_model(model_name, corpus, corpus_embs, split, "base", unique_id=unique_id)
        print(f"Evaluating Cobweb Basic")
        results.append(evaluate_retrieval("Cobweb Basic", queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb), top_k))
        print(f"--- Cobweb Basic Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)

    if 'Cobweb PCA + ICA' in get_benchmarks:
        print(f"Setting up PCA + ICA Cobweb...")
        cobweb_pca_ica = load_cobweb_model(model_name, corpus, pca_ica_corpus_embs, split, "pca_ica", unique_id=unique_id)
        print(f"Evaluating Cobweb PCA + ICA")
        results.append(evaluate_retrieval("Cobweb PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica), top_k))
        print(f"--- Cobweb PCA + ICA Metrics ---")
        print_metrics_table(results[-1], save_path=save_path)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run MS Marco Benchmark with configurable parameters")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--model_name", type=str, default="all-roberta-large-v1", help="Model name for embeddings")
    parser.add_argument("--subset_size", type=int, default=7500, help="Size of the corpus subset")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")
    parser.add_argument("--target_size", type=int, default=750, help="Number of query-target pairs")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k for retrieval evaluation")
    parser.add_argument("--compute", action="store_true", default=True, help="Whether to compute embeddings")
    parser.add_argument("--method", default="all", choices=["all", "extra", "cobweb"], help="Method to use for retrieval (all, extra, cobweb)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Handle config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Handle compute flag
    
    print(f"Running MS Marco benchmark with:")
    print(f"  Model: {args.model_name}")
    print(f"  Subset size: {args.subset_size}")
    print(f"  Split: {args.split}")
    print(f"  Target size: {args.target_size}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Compute: {args.compute}")
    print(f"  Method: {args.method}")
    
    results = run_msmarco_benchmark(
        model_name=args.model_name,
        subset_size=args.subset_size,
        split=args.split,
        target_size=args.target_size,
        top_k=args.top_k,
        compute=args.compute,
        method=args.method
    )