import os
import argparse
import json
import numpy as np
from random import sample as randsample, shuffle
from datasets import load_dataset

from src.utils.benchmark_utils import (
    generate_unique_id, load_or_compute_embeddings, load_or_save_sentences,
    load_cobweb_model, load_pca_ica_model, setup_faiss, retrieve_faiss,
    retrieve_cobweb_basic, evaluate_retrieval, print_metrics_table, get_results_path
)


def run_qqp_benchmark(model_name, subset_size=7500, split="validation", target_size=750, top_k=3, compute=True):
    print(f"--- Running QQP Benchmark (TOP_K={top_k}) ---")

    # Generate unique identifier for this run
    unique_id = generate_unique_id(
        model_name=model_name,
        dataset="qqp",
        split=split,
        subset_size=subset_size,
        target_size=target_size
    )
    print(f"Run ID: {unique_id}")

    corpus, queries, targets = None, None, None
    if compute:
        # Load QQP dataset and extract duplicates
        dataset = load_dataset("glue", "qqp", split=split)

        # Filter duplicates where label == 1
        duplicates = [ex for ex in dataset if ex["label"] == 1]

        shuffle(duplicates)
        sampled = randsample(duplicates, min(subset_size, len(duplicates)))
        queries = [ex["question1"] for ex in sampled[:target_size]]
        targets = [ex["question2"] for ex in sampled[:target_size]]
        corpus = [ex["question2"] for ex in sampled]

        print("Length of Corpus:", len(corpus))

    corpus_embs = load_or_compute_embeddings(corpus, model_name, "qqp_corpus", split, compute=compute, unique_id=unique_id)
    print(f"Corpus embeddings shape: {corpus_embs.shape}")
    queries_embs = load_or_compute_embeddings(queries, model_name, "qqp_queries", split, compute=compute, unique_id=unique_id)
    print(f"Queries embeddings shape: {queries_embs.shape}")
    targets = load_or_save_sentences(targets, model_name, "qqp_targets", split, compute=compute, unique_id=unique_id)
    corpus = load_or_save_sentences(corpus, model_name, "qqp_corpus", split, compute=compute, unique_id=unique_id)

    pca_ica_model = load_pca_ica_model(corpus_embs, model_name, "qqp_corpus", split, "general", target_dim=0.9, unique_id=unique_id)
    print(f"PCA/ICA model loaded: {pca_ica_model}")

    print(f"Starting PCA and ICA embeddings transformation...")
    pca_ica_corpus_embs = pca_ica_model.transform(corpus_embs)
    pca_ica_queries_embs = pca_ica_model.transform(queries_embs)
    print(f"PCA and ICA embeddings transformation completed.")

    # Setup retrieval methods
    results = []
    save_path = get_results_path(model_name, "qqp", split, unique_id)
    
    print(f"Setting up FAISS...")
    faiss_index = setup_faiss(corpus_embs)
    results.append(evaluate_retrieval("FAISS", queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_index, corpus), top_k))
    print(f"--- FAISS Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    print(f"Setting up FAISS with PCA + ICA...")
    faiss_pca_ica_index = setup_faiss(pca_ica_corpus_embs)
    results.append(evaluate_retrieval("FAISS PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_faiss(q, k, faiss_pca_ica_index, corpus), top_k))
    print(f"--- FAISS PCA + ICA Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    print(f"Setting up Basic Cobweb...")
    cobweb = load_cobweb_model(model_name, corpus, corpus_embs, split, "base", unique_id=unique_id)

    print(f"Evaluating Cobweb Basic")
    results.append(evaluate_retrieval("Cobweb Basic", queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb), top_k))
    print(f"--- Cobweb Basic Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    print(f"Setting up PCA + ICA Cobweb...")
    cobweb_pca_ica = load_cobweb_model(model_name, corpus, pca_ica_corpus_embs, split, "pca_ica", unique_id=unique_id)

    print(f"Evaluating Cobweb PCA + ICA")
    results.append(evaluate_retrieval("Cobweb PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica), top_k))
    print(f"--- Cobweb PCA + ICA Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run QQP Benchmark with configurable parameters")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--model_name", type=str, default="all-roberta-large-v1", help="Model name for embeddings")
    parser.add_argument("--subset_size", type=int, default=7500, help="Size of the corpus subset")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")
    parser.add_argument("--target_size", type=int, default=750, help="Number of query-target pairs")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k for retrieval evaluation")
    parser.add_argument("--compute", action="store_true", default=True, help="Whether to compute embeddings")
    parser.add_argument("--no_compute", action="store_true", help="Skip computing embeddings")
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
    compute = args.compute and not args.no_compute
    
    print(f"Running QQP benchmark with:")
    print(f"  Model: {args.model_name}")
    print(f"  Subset size: {args.subset_size}")
    print(f"  Split: {args.split}")
    print(f"  Target size: {args.target_size}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Compute: {compute}")
    
    results = run_qqp_benchmark(
        model_name=args.model_name,
        subset_size=args.subset_size,
        split=args.split,
        target_size=args.target_size,
        top_k=args.top_k,
        compute=compute
    )
    for res in results:
        print_metrics_table(res)