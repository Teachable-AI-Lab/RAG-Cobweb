import glob
import re
from collections import defaultdict

def parse_filename_info(filename):
    """Extract corpus size from filename."""
    corpus_match = re.search(r'c(\d+)', filename)
    corpus_size = int(corpus_match.group(1)) if corpus_match else None
    return corpus_size

def parse_metrics_txt(filepath):
    corpus_size = parse_filename_info(filepath)
    with open(filepath, "r") as f:
        content = f.read()

    results = defaultdict(dict)
    blocks = content.strip().split("\n\n")
    for block in blocks:
        match_method = re.search(r"--- Metrics for (.+) ---", block)
        if not match_method:
            continue
        method = match_method.group(1).strip()

        match_time = re.search(r"Avg Latency:\s*([\d.]+)\s*ms", block)
        latency = float(match_time.group(1)) if match_time else None

        if corpus_size is not None:
            results[method][corpus_size] = latency
    return results

def merge_results(all_results, new_results):
    for method, runs in new_results.items():
        if method not in all_results:
            all_results[method] = {}
        all_results[method].update(runs)

def generate_runtime_table(results, methods):
    # Get sorted corpus sizes
    all_corpus_sizes = sorted({size for runs in results.values() for size in runs})
    table = "\\begin{table}[h!]\n\\centering\n\\caption{Average Latency Across Corpus Sizes}\n"
    table += "\\begin{tabular}{l" + "c" * len(all_corpus_sizes) + "}\n\\hline\n"
    headers = ["\\textbf{Method}"] + [str(c) for c in all_corpus_sizes]
    table += " & ".join(headers) + " \\\\\n\\hline\n"
    for method in methods:
        row = [method]
        for size in all_corpus_sizes:
            lat = results.get(method, {}).get(size, "--")
            row.append(f"{lat:.2f}" if isinstance(lat, float) else "--")
        table += " & ".join(row) + " \\\\\n"
    table += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return table

def create_save_runtime_table():
    all_results = {}
    # Automatically process all files with 'all-roberta-large-v1' in their names from outputs/qqp
    for file in glob.glob("outputs/msmarco/*gtr-t5-large*.txt"):
        merge_results(all_results, parse_metrics_txt(file))

    runtime_methods = ["FAISS", "Cobweb Basic", "Cobweb Fast", "Cobweb PCA + ICA", "Cobweb PCA + ICA Fast"]
    latex_table = generate_runtime_table(all_results, runtime_methods)
    print(latex_table)
    # Save the LaTeX table to a file
    with open("runtime_table.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    create_save_runtime_table()
