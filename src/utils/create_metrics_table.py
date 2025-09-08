import glob
import re
import sys
from collections import defaultdict

def parse_model_name(filename):
    """
    Extract model name from filename.
    Assumes something like 'benchmark_all-roberta-large-v1_validation...'
    """
    match = re.search(r'benchmark_([^_]+)_', filename)
    if match:
        return match.group(1)
    return "UnknownModel"

def parse_metrics_file(filepath):
    """
    Parse one metrics .txt file into {model_name: {method: {metrics: {k: (recall, mrr, ndcg)}}}}
    """
    model_name = parse_model_name(filepath)
    with open(filepath, "r") as f:
        content = f.read()

    model_results = defaultdict(lambda: {"metrics": {}})
    blocks = content.strip().split("\n\n")
    for block in blocks:
        match_method = re.search(r"--- Metrics for (.+) ---", block)
        if not match_method:
            continue
        method = match_method.group(1).strip()

        # Parse metrics table rows
        for line in block.split("\n"):
            if line.strip().startswith("| @"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                k = int(parts[0].replace("@", ""))
                recall, mrr, ndcg = parts[1:4]
                if "metrics" not in model_results[method]:
                    model_results[method]["metrics"] = {}
                model_results[method]["metrics"][k] = (recall, mrr, ndcg)

    return model_name, dict(model_results)

def parse_corpus_size(filename):
    """Extract corpus size from filename (e.g., c10000)."""
    match = re.search(r'_c(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def regroup_results_by_corpus(all_results, file_map):
    """
    Reorganize all_results to group by corpus size instead of model.
    Returns: corpus_results[corpus_size][method][model] = metrics_dict
    file_map: {model: filename}
    """
    corpus_results = defaultdict(lambda: defaultdict(dict))
    for model, methods in all_results.items():
        filename = file_map.get(model, "")
        corpus_size = parse_corpus_size(filename)
        if corpus_size is None:
            continue
        for method, method_data in methods.items():
            corpus_results[corpus_size][method][model] = method_data.get("metrics", {})
    return corpus_results

def generate_grouped_metrics_table(k, results, methods, models, selected_metrics=None):
    """
    Generate LaTeX table for metrics at given k with groups for each model.
    results[model][method]["metrics"][k] = (recall, mrr, ndcg)
    selected_metrics: list of metric names to include (e.g., ['Recall', 'MRR', 'nDCG'])
    """
    if selected_metrics is None:
        selected_metrics = ['Recall', 'MRR', 'nDCG']
    
    metric_indices = {'Recall': 0, 'MRR': 1, 'nDCG': 2}
    num_metrics = len(selected_metrics)
    
    table = "\\begin{table}[h!]\n\\centering\n"
    table += f"\\caption{{Metrics @k={k} for Different Models}}\n"
    table += "\\begin{tabular}{l" + "c" * num_metrics * len(models) + "}\n\\hline\n"
    
    # Model headers
    table += " & " + " & ".join([f"\\multicolumn{{{num_metrics}}}{{c}}{{{model}}}" for model in models]) + " \\\\\n"
    # Metric headers
    table += "\\textbf{Method} & " + " & ".join([" & ".join(selected_metrics)] * len(models)) + " \\\\\n\\hline\n"

    for method in methods:
        row = [method]
        for model in models:
            metrics_dict = results.get(model, {}).get(method, {}).get("metrics", {})
            if k in metrics_dict:
                recall, mrr, ndcg = metrics_dict[k]
                all_metrics = [recall, mrr, ndcg]
                row.extend([all_metrics[metric_indices[metric]] for metric in selected_metrics])
            else:
                row.extend(["--"] * num_metrics)
        table += " & ".join(row) + " \\\\\n"

    table += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return table

def generate_grouped_metrics_table_by_corpus(k, corpus_results, methods, models, corpus_sizes, selected_metrics=None):
    """
    Generate LaTeX table for metrics at given k with groups for each corpus size.
    corpus_results[corpus_size][method][model] = metrics_dict
    selected_metrics: list of metric names to include (e.g., ['Recall', 'MRR', 'nDCG'])
    """
    if selected_metrics is None:
        selected_metrics = ['Recall', 'MRR', 'nDCG']
    
    metric_indices = {'Recall': 0, 'MRR': 1, 'nDCG': 2}
    num_metrics = len(selected_metrics)
    
    table = "\\begin{table}[h!]\n\\centering\n"
    table += f"\\caption{{Metrics @k={k} for Different Corpus Sizes}}\n"
    table += "\\begin{tabular}{l" + "c" * num_metrics * len(corpus_sizes) + "}\n\\hline\n"
    # Corpus size headers
    table += " & " + " & ".join([f"\\multicolumn{{{num_metrics}}}{{c}}{{c={size}}}" for size in corpus_sizes]) + " \\\\\n"
    # Metric headers
    table += "\\textbf{Method} & " + " & ".join([" & ".join(selected_metrics)] * len(corpus_sizes)) + " \\\\\n\\hline\n"
    for method in methods:
        row = [method]
        for size in corpus_sizes:
            # Aggregate over models (show first model found, or average if you want)
            metrics_dicts = corpus_results.get(size, {}).get(method, {})
            # Pick first model's metrics for this method and corpus size
            found = False
            for model in models:
                metrics = metrics_dicts.get(model, {})
                if k in metrics:
                    recall, mrr, ndcg = metrics[k]
                    all_metrics = [recall, mrr, ndcg]
                    row.extend([all_metrics[metric_indices[metric]] for metric in selected_metrics])
                    found = True
                    break
            if not found:
                row.extend(["--"] * num_metrics)
        table += " & ".join(row) + " \\\\\n"
    table += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return table


def run_encoder_table(data_name):
    all_results = defaultdict(dict)  # all_results[model][method]...
    file_map = {}  # model -> filename
    methods = ["FAISS", "FAISS PCA + ICA", "Cobweb Basic", "Cobweb PCA + ICA", "Cobweb PCA + ICA Fast"]
    if data_name == "qqp":
        gpt2 = "outputs/qqp/benchmark_openai-community-gpt2_validation_openai-community-gpt2_qqp_validation_c10000_t1000_k10_2883a1cb.txt"
        roberta = "outputs/qqp/benchmark_all-roberta-large-v1_validation_all-roberta-large-v1_qqp_validation_c10000_t1000_k20_d21a8956.txt"
        t5 = "outputs/qqp/benchmark_gtr-t5-large_validation_gtr-t5-large_qqp_validation_c10000_t1000_k20_401c8152.txt"
        for file in [gpt2, roberta, t5]:
            model_name, parsed = parse_metrics_file(file)
            all_results[model_name].update(parsed)
            file_map[model_name] = file
    elif data_name == "ms_marco":
        gpt2 = "outputs/msmarco/benchmark_openai-community-gpt2_validation_openai-community-gpt2_msmarco_validation_c10000_t1000_k10_748c3d17.txt"
        roberta = "outputs/msmarco/benchmark_all-roberta-large-v1_validation_all-roberta-large-v1_msmarco_validation_c10000_t1000_k20_df03abf8.txt"
        t5 = "outputs/msmarco/benchmark_gtr-t5-large_validation_gtr-t5-large_msmarco_validation_c10000_t1000_k20_024357d7.txt"
        for file in [gpt2, roberta, t5]:
            model_name, parsed = parse_metrics_file(file)
            all_results[model_name].update(parsed)
            file_map[model_name] = file
    models = sorted(all_results.keys())
    k_lst = [5,10]
    for k in k_lst:
        latex_table = generate_grouped_metrics_table(k, all_results, methods, models, selected_metrics)
        metrics_suffix = "_".join([metric.lower() for metric in selected_metrics])
        filename = f"{data_name}_metrics_table_k{k}_by_model_{metrics_suffix}.tex"
        with open(filename, "w") as f:
            f.write(latex_table)
        print(f"Generated table with {', '.join(selected_metrics)} metrics: {filename}")

def run_scale_table(data_name, model_name):
    all_results = defaultdict(dict)  # all_results[model][method]...
    file_map = {}  # model -> filename
    methods = ["FAISS", "Cobweb PCA + ICA", "Cobweb PCA + ICA Fast"]
    selected_metrics = ['Recall', 'MRR']

    for file in glob.glob(f"outputs/{data_name}/benchmark_{model_name}_*.txt"):
        _, parsed = parse_metrics_file(file)
        corpus_size = parse_corpus_size(file)
        all_results[corpus_size].update(parsed)
        file_map[corpus_size] = file
    models = sorted(all_results.keys())
    if 7500 in models:  # Remove 7500 if it exists
        models.remove(7500)
    if 1000 in models:  # Remove 1000 if it exists
        models.remove(1000)
    data_name = (model_name + "-" + data_name)

    # Option 1: Generate table grouped by model (original)
    k_lst = [5,10]
    for k in k_lst:
        latex_table = generate_grouped_metrics_table(k, all_results, methods, models, selected_metrics)
        metrics_suffix = "_".join([metric.lower() for metric in selected_metrics])
        filename = f"{data_name}_metrics_table_k{k}_by_model_{metrics_suffix}.tex"
        with open(filename, "w") as f:
            f.write(latex_table)
        print(f"Generated table with {', '.join(selected_metrics)} metrics: {filename}")


if __name__ == "__main__":
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Parse metrics from command line: python create_metrics_table.py Recall MRR
        available_metrics = ['Recall', 'MRR', 'nDCG']
        selected_metrics = []
        for arg in sys.argv[1:]:
            if arg in available_metrics:
                selected_metrics.append(arg)
            else:
                print(f"Warning: '{arg}' is not a valid metric. Available metrics: {available_metrics}")
        
        if not selected_metrics:
            print("No valid metrics specified. Using all metrics.")
            selected_metrics = ['Recall', 'MRR', 'nDCG']
        else:
            print(f"Selected metrics: {selected_metrics}")
    else:
        # Default: use all metrics
        selected_metrics = ['Recall', 'MRR', 'nDCG']  # Change this to select specific metrics
    
    # Gather all results
    option = 2
    
    # data_name = "all-roberta-large-v1"
    if option == 1:
        for data_name in ["qqp", "msmarco"]:
            run_encoder_table(data_name)
    elif option == 2:
        for data_name in ["qqp", "msmarco"]:
            for model_name in ["gtr-t5-large", "all-roberta-large-v1"]:
                run_scale_table(data_name, model_name)

    # List of methods and models to include
