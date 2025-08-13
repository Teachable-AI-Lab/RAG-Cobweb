from random import sample as randsample, shuffle
from datasets import load_dataset
from typing import List, Tuple

from src.benchmarks.base_benchmark import BaseBenchmark


class QQPBenchmark(BaseBenchmark):
    """QQP dataset benchmark implementation."""
    
    def __init__(self):
        super().__init__("qqp")
    
    def load_dataset(self, subset_size: int = 7500, split: str = "validation", 
                    target_size: int = 750) -> Tuple[List[str], List[str], List[str]]:
        """Load QQP dataset and extract duplicates."""
        # Load QQP dataset and extract duplicates
        dataset = load_dataset("glue", "qqp", split=split)

        # Filter duplicates where label == 1
        duplicates = []
        extra = []
        for ex in dataset:
            if ex["label"] == 1:
                duplicates.append(ex)
            else:
                extra.append(ex["question2"])

        shuffle(duplicates)
        sampled = randsample(duplicates, min(subset_size, len(duplicates)))
        queries = [ex["question1"] for ex in sampled[:target_size]]
        targets = [ex["question2"] for ex in sampled[:target_size]]
        corpus = [ex["question2"] for ex in sampled]
        if len(corpus) < subset_size:
            corpus += randsample(extra, min(subset_size - len(corpus), len(extra)))
        return corpus, queries, targets


def run_qqp_benchmark(model_name: str, subset_size: int = 7500, split: str = "validation", 
                     target_size: int = 750, top_k: int = 3, compute: bool = True, 
                     method: str = 'all') -> List[dict]:
    """Run QQP benchmark using the new consolidated approach."""
    benchmark = QQPBenchmark()
    return benchmark.run_benchmark(
        model_name=model_name,
        subset_size=subset_size,
        split=split,
        target_size=target_size,
        top_k=top_k,
        compute=compute,
        method=method,
        target_dim=0.96,
        include_cobweb_fast=True
    )


def parse_args():
    """Parse command line arguments for QQP benchmark."""
    parser = BaseBenchmark.create_argument_parser("Run QQP Benchmark with configurable parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Handle config file if provided
    args = BaseBenchmark.handle_config_and_args(args)
    
    # Print run information
    BaseBenchmark.print_run_info(args, "qqp")
    
    results = run_qqp_benchmark(
        model_name=args.model_name,
        subset_size=args.subset_size,
        split=args.split,
        target_size=args.target_size,
        top_k=args.top_k,
        compute=args.compute,
        method=args.method
    )