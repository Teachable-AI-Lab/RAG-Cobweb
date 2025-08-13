# Benchmark Experiment Runner

This system allows you to run multiple benchmark experiments with different configurations without manually editing files.

## Quick Start

### Option 1: Use the Master Script (Recommended)

Submit the master script to SLURM, which will then submit individual experiment jobs:

```bash
# Run all experiments defined in configs/experiments.json
sbatch scripts/submit_experiments.sh

# Run specific experiments only
sbatch scripts/submit_experiments.sh --experiment qqp_roberta_small ms_marco_dpr

# Dry run to see what would be executed
sbatch scripts/submit_experiments.sh --dry_run
```

### Option 2: Run Individual Scripts

```bash
# Run with default config
sbatch scripts/benchmarks/qqp.sh

# Run with custom config
sbatch scripts/benchmarks/qqp.sh configs/benchmarks/qqp_t5.json

# Run with command line arguments (no config file)
sbatch --wrap "cd /nethome/$USER/flash/RAG-Cobweb && python src/benchmarks/qqp_dataset.py --model_name google-t5/t5-base --top_k 5"
```

### Option 3: Direct Python Execution

```bash
# With config file
python src/benchmarks/qqp_dataset.py --config configs/benchmarks/qqp_default.json

# With command line arguments
python src/benchmarks/qqp_dataset.py --model_name all-roberta-large-v1 --subset_size 1000 --top_k 10

# Mix of config file and overrides
python src/benchmarks/qqp_dataset.py --config configs/benchmarks/qqp_default.json --top_k 20
```

## Configuration Files

### Experiment Configuration (`configs/experiments.json`)

Defines multiple experiments to run:

```json
{
    "experiments": [
        {
            "name": "experiment_name",
            "dataset": "qqp|ms_marco", 
            "config": "path/to/config.json",
            "slurm_options": {
                "time": "02:00:00",
                "mem": "16G",
                "cpus_per_task": 4
            }
        }
    ],
    "default_slurm_options": {
        "partition": "tail-lab",
        "nodes": 1,
        "ntasks_per_node": 1,
        "cpus_per_task": 4,
        "mem": "16G", 
        "qos": "short",
        "exclude": "clippy"
    }
}
```

### Individual Benchmark Configurations

Located in `configs/benchmarks/`:

- `qqp_default.json` - Default QQP benchmark settings
- `qqp_t5.json` - QQP with T5 model
- `ms_marco_default.json` - Default MS Marco settings
- `ms_marco_dpr.json` - MS Marco with DPR model

Example:
```json
{
    "model_name": "all-roberta-large-v1",
    "subset_size": 1000,
    "split": "validation", 
    "target_size": 100,
    "top_k": 10,
    "compute": true
}
```

## Command Line Arguments

Both `qqp_dataset.py` and `ms_marco_dataset.py` support:

- `--config`: Path to JSON config file
- `--model_name`: Model name for embeddings
- `--subset_size`: Size of the corpus subset  
- `--split`: Dataset split to use
- `--target_size`: Number of query-target pairs
- `--top_k`: Top-k for retrieval evaluation
- `--compute`: Compute embeddings (default: True)
- `--no_compute`: Skip computing embeddings

## Monitoring Jobs

```bash
# Check your running jobs
squeue -u $USER

# Check specific jobs
squeue -j job_id1,job_id2

# Check job output in real time
tail -f /nethome/$USER/flash/slurm_outputs/job_name.out
```

## Adding New Experiments

1. Create a new config file in `configs/benchmarks/qqp|ms_marco`
2. Add the experiment to `configs/experiments_qqp|ms_marco.json`
3. Run with the master script
