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
    evaluate_retrieval, print_metrics_table, get_results_path
)


def run_msmarco_benchmark(model_name="facebook/dpr-question_encoder-single-nq-base", subset_size=7500, split="validation", target_size=750, top_k=3, compute=True):
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

    print(f"Setting up PCA + ICA Cobweb...")
    cobweb_pca_ica = load_cobweb_model(query_model_name, corpus, pca_ica_corpus_embs, split, "pca_ica", unique_id=unique_id)
    results.append(evaluate_retrieval("Cobweb PCA + ICA", pca_ica_queries_embs, targets, lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica), top_k))
    print(f"--- Cobweb PCA + ICA Metrics ---")
    print_metrics_table(results[-1], save_path=save_path)

    cobweb_pca_ica.visualize_subtrees("outputs/visualizations_ms_marco")

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
    
    print(f"Running MS Marco benchmark with:")
    print(f"  Model: {args.model_name}")
    print(f"  Subset size: {args.subset_size}")
    print(f"  Split: {args.split}")
    print(f"  Target size: {args.target_size}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Compute: {compute}")
    
    results = run_msmarco_benchmark(
        model_name=args.model_name,
        subset_size=args.subset_size,
        split=args.split,
        target_size=args.target_size,
        top_k=args.top_k,
        compute=compute
    )
    for res in results:
        print_metrics_table(res)