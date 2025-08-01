"""
Base benchmark class for consolidating common retrieval benchmarking functionality.
Provides a template that dataset-specific benchmarks can inherit from.
"""
import os
import argparse
import json
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

from src.utils.benchmark_utils import (
    generate_unique_id, load_or_compute_embeddings, load_or_compute_dpr_embeddings,
    load_or_save_sentences, load_cobweb_model, load_pca_ica_model,
    setup_faiss, retrieve_faiss, retrieve_cobweb_basic,
    evaluate_retrieval, print_metrics_table, get_results_path,
    setup_hnswlib, retrieve_hnswlib
)


class BaseBenchmark(ABC):
    """Base class for retrieval benchmarks."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
    
    @abstractmethod
    def load_dataset(self, subset_size: int, split: str, target_size: int) -> Tuple[List[str], List[str], List[str]]:
        """
        Load dataset and return corpus, queries, targets.
        
        Args:
            subset_size: Size of the corpus subset
            split: Dataset split to use
            target_size: Number of query-target pairs
            
        Returns:
            Tuple of (corpus, queries, targets)
        """
        pass
    
    def get_benchmark_list(self, method: str = "all") -> List[str]:
        """Get list of benchmark methods to run."""
        if method == "all":
            return ['FAISS', 'FAISS PCA + ICA', 'Cobweb Basic', 'Cobweb PCA + ICA']
        elif method == "extra":
            return ['FAISS', 'FAISS PCA + ICA', 'FAISS L2', 'FAISS L2 PCA + ICA', 
                   'HNSWLib', 'HNSWLib PCA + ICA', 'Cobweb Basic', 'Cobweb PCA + ICA']
        elif method == "cobweb":
            return ['Cobweb Basic', 'Cobweb PCA + ICA']
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def setup_embeddings(self, corpus: List[str], queries: List[str], targets: List[str],
                        model_name: str, split: str, compute: bool, unique_id: str,
                        is_dpr_model: bool = False) -> Dict[str, Any]:
        """Setup embeddings for corpus and queries."""
        if is_dpr_model:
            # DPR model names
            query_model_name = model_name
            passage_model_name = model_name.replace("question_encoder", "ctx_encoder")
            print(f"Using Query Model: {query_model_name}")
            print(f"Using Passage Model: {passage_model_name}")
            
            corpus_embs = load_or_compute_dpr_embeddings(
                corpus, "passage", passage_model_name, 
                f"{self.dataset_name}_corpus", split, compute=compute, unique_id=unique_id
            )
            queries_embs = load_or_compute_dpr_embeddings(
                queries, "query", query_model_name,
                f"{self.dataset_name}_queries", split, compute=compute, unique_id=unique_id
            )
            
            # Save sentences with appropriate model names
            targets = load_or_save_sentences(targets, query_model_name, f"{self.dataset_name}_targets", split, compute=compute, unique_id=unique_id)
            corpus = load_or_save_sentences(corpus, passage_model_name, f"{self.dataset_name}_corpus", split, compute=compute, unique_id=unique_id)
        else:
            # Use the same model for both queries and passages
            corpus_embs = load_or_compute_embeddings(corpus, model_name, f"{self.dataset_name}_corpus", split, compute=compute, unique_id=unique_id)
            queries_embs = load_or_compute_embeddings(queries, model_name, f"{self.dataset_name}_queries", split, compute=compute, unique_id=unique_id)
            
            targets = load_or_save_sentences(targets, model_name, f"{self.dataset_name}_targets", split, compute=compute, unique_id=unique_id)
            corpus = load_or_save_sentences(corpus, model_name, f"{self.dataset_name}_corpus", split, compute=compute, unique_id=unique_id)
        
        print(f"Corpus embeddings shape: {corpus_embs.shape}")
        print(f"Queries embeddings shape: {queries_embs.shape}")
        
        return {
            'corpus_embs': corpus_embs,
            'queries_embs': queries_embs,
            'corpus': corpus,
            'targets': targets
        }
    
    def setup_pca_ica_models(self, corpus_embs: np.ndarray, queries_embs: np.ndarray,
                            model_name: str, split: str, unique_id: str,
                            target_dim: float = 0.96, compute: bool = False) -> Dict[str, Any]:
        """Setup PCA+ICA models and transform embeddings."""
        # Determine input for PCA+ICA fitting based on dataset
        if self.dataset_name == "msmarco":
            # For MS Marco, fit on combined embeddings
            pca_ica_input = np.vstack((corpus_embs, queries_embs))
        else:
            # For QQP and others, fit only on corpus
            pca_ica_input = corpus_embs
            
        pca_ica_model = load_pca_ica_model(
            pca_ica_input, model_name, self.dataset_name, split, 
            "general", target_dim=target_dim, unique_id=unique_id, compute=compute
        )
        print(f"PCA + ICA model loaded: {pca_ica_model}")
        
        print(f"Starting PCA and ICA embeddings transformation...")
        pca_ica_corpus_embs = pca_ica_model.transform(corpus_embs)
        pca_ica_queries_embs = pca_ica_model.transform(queries_embs)
        print(f"PCA and ICA embeddings transformation completed.")
        
        return {
            'pca_ica_model': pca_ica_model,
            'pca_ica_corpus_embs': pca_ica_corpus_embs,
            'pca_ica_queries_embs': pca_ica_queries_embs
        }
    
    def run_benchmark_methods(self, embeddings: Dict[str, Any], pca_ica_data: Dict[str, Any],
                             model_name: str, split: str, unique_id: str, top_k: int,
                             method: str, include_cobweb_fast: bool = True) -> List[Dict[str, Any]]:
        """Run all benchmark methods and return results."""
        corpus_embs = embeddings['corpus_embs']
        queries_embs = embeddings['queries_embs']
        corpus = embeddings['corpus']
        targets = embeddings['targets']
        
        pca_ica_corpus_embs = pca_ica_data['pca_ica_corpus_embs']
        pca_ica_queries_embs = pca_ica_data['pca_ica_queries_embs']
        
        results = []
        save_path = get_results_path(model_name, self.dataset_name, split, unique_id)
        get_benchmarks = self.get_benchmark_list(method)
        
        # FAISS benchmarks
        if 'FAISS' in get_benchmarks:
            print(f"Setting up FAISS...")
            faiss_index = setup_faiss(corpus_embs)
            results.append(evaluate_retrieval("FAISS", queries_embs, targets, 
                                            lambda q, k: retrieve_faiss(q, k, faiss_index, corpus), top_k))
            print(f"--- FAISS Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
        if 'FAISS PCA + ICA' in get_benchmarks:
            print(f"Setting up FAISS with PCA + ICA...")
            faiss_pca_ica_index = setup_faiss(pca_ica_corpus_embs)
            results.append(evaluate_retrieval("FAISS PCA + ICA", pca_ica_queries_embs, targets,
                                            lambda q, k: retrieve_faiss(q, k, faiss_pca_ica_index, corpus), top_k))
            print(f"--- FAISS PCA + ICA Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
        if 'FAISS L2' in get_benchmarks:
            print(f"Setting up FAISS with L2 distance...")
            faiss_l2_index = setup_faiss(corpus_embs, index_type='l2')
            results.append(evaluate_retrieval("FAISS L2", queries_embs, targets,
                                            lambda q, k: retrieve_faiss(q, k, faiss_l2_index, corpus), top_k))
            print(f"--- FAISS L2 Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
        if 'FAISS L2 PCA + ICA' in get_benchmarks:
            print(f"Setting up FAISS with L2 distance and PCA + ICA...")
            faiss_l2_pca_ica_index = setup_faiss(pca_ica_corpus_embs, index_type='l2')
            results.append(evaluate_retrieval("FAISS L2 PCA + ICA", pca_ica_queries_embs, targets,
                                            lambda q, k: retrieve_faiss(q, k, faiss_l2_pca_ica_index, corpus), top_k))
            print(f"--- FAISS L2 PCA + ICA Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
        # HNSWLib benchmarks
        if 'HNSWLib' in get_benchmarks:
            print(f"Setting up HNSWLib...")
            hnswlib_index = setup_hnswlib(corpus_embs)
            results.append(evaluate_retrieval("HNSWLib", queries_embs, targets,
                                            lambda q, k: retrieve_hnswlib(q, k, hnswlib_index, corpus), top_k))
            print(f"--- HNSWLib Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
        if 'HNSWLib PCA + ICA' in get_benchmarks:
            print(f"Setting up HNSWLib with PCA + ICA...")
            hnswlib_pca_ica_index = setup_hnswlib(pca_ica_corpus_embs)
            results.append(evaluate_retrieval("HNSWLib PCA + ICA", pca_ica_queries_embs, targets,
                                            lambda q, k: retrieve_hnswlib(q, k, hnswlib_pca_ica_index, corpus), top_k))
            print(f"--- HNSWLib PCA + ICA Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
        # Cobweb benchmarks
        if 'Cobweb Basic' in get_benchmarks:
            print(f"Setting up Basic Cobweb...")
            cobweb = load_cobweb_model(model_name, corpus, corpus_embs, split, "base", unique_id=unique_id)
            print(f"Evaluating Cobweb Basic")
            results.append(evaluate_retrieval("Cobweb Basic", queries_embs, targets,
                                            lambda q, k: retrieve_cobweb_basic(q, k, cobweb), top_k))
            print(f"--- Cobweb Basic Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
        if 'Cobweb PCA + ICA' in get_benchmarks:
            print(f"Setting up PCA + ICA Cobweb...")
            cobweb_pca_ica = load_cobweb_model(model_name, corpus, pca_ica_corpus_embs, split, "pca_ica", unique_id=unique_id)
            print(f"Evaluating Cobweb PCA + ICA")
            results.append(evaluate_retrieval("Cobweb PCA + ICA", pca_ica_queries_embs, targets,
                                            lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica), top_k))
            print(f"--- Cobweb PCA + ICA Metrics ---")
            print_metrics_table(results[-1], save_path=save_path)
            
            # Add fast Cobweb evaluation if requested
            if include_cobweb_fast:
                print(f"--- Evaluating PCA + ICA Fast ---")
                t = time.time()
                cobweb_pca_ica.build_prediction_index()
                print(f"--- Cobweb PCA + ICA Fast Index Built in {time.time() - t:.2f} seconds ---")
                results.append(evaluate_retrieval("Cobweb PCA + ICA Fast", pca_ica_queries_embs, targets,
                                                lambda q, k: retrieve_cobweb_basic(q, k, cobweb_pca_ica, True), top_k))
                print(f"--- Cobweb PCA + ICA Fast Metrics ---")
                print_metrics_table(results[-1], save_path=save_path)
        
        return results
    
    def run_benchmark(self, model_name: str, subset_size: int = 7500, split: str = "validation",
                     target_size: int = 750, top_k: int = 3, compute: bool = True,
                     method: str = 'all', **kwargs) -> List[Dict[str, Any]]:
        """
        Run the complete benchmark pipeline.
        
        Args:
            model_name: Name of the model to use
            subset_size: Size of the corpus subset
            split: Dataset split to use
            target_size: Number of query-target pairs
            top_k: Top-k for retrieval evaluation
            compute: Whether to compute embeddings
            method: Which benchmark methods to run
            **kwargs: Additional dataset-specific parameters
            
        Returns:
            List of benchmark results
        """
        print(f"--- Running {self.dataset_name.upper()} Benchmark (TOP_K={top_k}) ---")
        
        # Check if this is a DPR model
        is_dpr_model = "dpr-" in model_name and ("question_encoder" in model_name or "ctx_encoder" in model_name)
        if is_dpr_model:
            print(f"Using DPR model: {model_name}")
        else:
            print(f"Using same model for both queries and passages: {model_name}")
        
        # Generate unique identifier for this run
        unique_id_params = {
            'model_name': model_name,
            'dataset': self.dataset_name,
            'split': split,
            'subset_size': subset_size,
            'target_size': target_size,
            'top_k': top_k,
            **kwargs
        }
        unique_id = generate_unique_id(**unique_id_params)
        print(f"Run ID: {unique_id}")
        
        # Load dataset
        corpus, queries, targets = None, None, None
        if compute:
            print(f"Loading {self.dataset_name.upper()} dataset...")
            corpus, queries, targets = self.load_dataset(subset_size, split, target_size)
            print(f"Loaded {len(corpus)} corpus entries, {len(queries)} queries, and {len(targets)} targets.")
        
        # Setup embeddings
        embeddings = self.setup_embeddings(corpus, queries, targets, model_name, split, compute, unique_id, is_dpr_model)
        
        # Setup PCA+ICA models
        target_dim = kwargs.get('target_dim', 0.96)
        pca_ica_data = self.setup_pca_ica_models(
            embeddings['corpus_embs'], embeddings['queries_embs'],
            model_name, split, unique_id, target_dim, compute
        )
        
        # Run benchmarks
        include_cobweb_fast = kwargs.get('include_cobweb_fast', False)
        results = self.run_benchmark_methods(
            embeddings, pca_ica_data, model_name, split, unique_id, 
            top_k, method, include_cobweb_fast
        )
        
        return results
    
    @staticmethod
    def create_argument_parser(description: str) -> argparse.ArgumentParser:
        """Create a standard argument parser for benchmarks."""
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--config", type=str, help="Path to JSON config file")
        parser.add_argument("--model_name", type=str, default="all-roberta-large-v1", help="Model name for embeddings")
        parser.add_argument("--subset_size", type=int, default=7500, help="Size of the corpus subset")
        parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")
        parser.add_argument("--target_size", type=int, default=750, help="Number of query-target pairs")
        parser.add_argument("--top_k", type=int, default=3, help="Top-k for retrieval evaluation")
        parser.add_argument("--compute", action="store_true", default=True, help="Whether to compute embeddings")
        parser.add_argument("--method", type=str, default="all", choices=["all", "extra", "cobweb"], help="Method to use for retrieval")
        return parser
    
    @staticmethod
    def handle_config_and_args(args: argparse.Namespace) -> argparse.Namespace:
        """Handle config file loading and argument override."""
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
            # Override args with config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        return args
    
    @staticmethod
    def print_run_info(args: argparse.Namespace, dataset_name: str):
        """Print run information."""
        print(f"Running {dataset_name.upper()} benchmark with:")
        print(f"  Model: {args.model_name}")
        print(f"  Subset size: {args.subset_size}")
        print(f"  Split: {args.split}")
        print(f"  Target size: {args.target_size}")
        print(f"  Top-k: {args.top_k}")
        print(f"  Compute: {args.compute}")
        print(f"  Method: {args.method}")
