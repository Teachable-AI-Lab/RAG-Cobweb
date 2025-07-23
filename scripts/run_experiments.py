#!/usr/bin/env python3
"""
Master script to run multiple benchmark experiments with different configurations.
"""
import os
import json
import argparse
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple benchmark experiments")
    parser.add_argument("--config", type=str, default="configs/experiments.json", 
                       help="Path to experiments configuration file")
    parser.add_argument("--experiment", type=str, nargs="+", 
                       help="Specific experiment names to run (run all if not specified)")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Print commands without executing them")
    parser.add_argument("--sequential", action="store_true",
                       help="Run experiments sequentially instead of submitting all at once")
    return parser.parse_args()


def create_slurm_script(experiment, config, base_dir):
    """Create a SLURM script for a specific experiment."""
    exp_name = experiment["name"]
    dataset = experiment["dataset"]
    config_path = experiment["config"]
    slurm_opts = experiment.get("slurm_options", {})
    
    # Merge with default SLURM options
    default_opts = config.get("default_slurm_options", {})
    merged_opts = {**default_opts, **slurm_opts}
    
    # Determine the Python script based on dataset
    if dataset == "qqp":
        python_script = "src/benchmarks/qqp_dataset.py"
    elif dataset == "ms_marco":
        python_script = "src/benchmarks/ms_marco_dataset.py"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=/nethome/agupta886/flash/slurm_outputs/{exp_name}.out
#SBATCH --error=/nethome/agupta886/flash/slurm_errors/{exp_name}.err
"""
    
    # Add SLURM options
    for key, value in merged_opts.items():
        if key == "mem":
            script_content += f"#SBATCH --mem={value}\n"
        elif key == "time":
            script_content += f"#SBATCH --time={value}\n"
        elif key == "partition":
            script_content += f"#SBATCH --partition={value}\n"
        elif key == "nodes":
            script_content += f"#SBATCH --nodes={value}\n"
        elif key == "ntasks_per_node":
            script_content += f"#SBATCH --ntasks-per-node={value}\n"
        elif key == "cpus_per_task":
            script_content += f"#SBATCH --cpus-per-task={value}\n"
        elif key == "qos":
            script_content += f"#SBATCH --qos={value}\n"
        elif key == "exclude":
            script_content += f"#SBATCH --exclude={value}\n"
        else:
            script_content += f"#SBATCH --{key.replace('_', '-')}={value}\n"
    
    script_content += f"""
export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd {base_dir}
export PYTHONPATH=$(pwd)

echo "Starting experiment: {exp_name}"
echo "Dataset: {dataset}"
echo "Config: {config_path}"
echo "Time: $(date)"

srun python {python_script} --config {config_path}

echo "Experiment completed: {exp_name}"
echo "Time: $(date)"
"""
    
    return script_content


def main():
    args = parse_args()
    
    # Get base directory (should be the RAG-Cobweb root directory)
    base_dir = Path(__file__).parent.parent.absolute()
    
    # Load experiments configuration
    config_path = base_dir / args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    experiments = config["experiments"]
    
    # Filter experiments if specific ones are requested
    if args.experiment:
        experiments = [exp for exp in experiments if exp["name"] in args.experiment]
        if not experiments:
            print(f"No experiments found matching: {args.experiment}")
            return
    
    print(f"Found {len(experiments)} experiments to run:")
    for exp in experiments:
        print(f"  - {exp['name']} ({exp['dataset']})")
    
    if args.dry_run:
        print("\nDry run mode - showing what would be executed:")
    
    # Create and submit jobs
    submitted_jobs = []
    
    for experiment in experiments:
        exp_name = experiment["name"]
        
        # Create SLURM script
        script_content = create_slurm_script(experiment, config, base_dir)
        
        if args.dry_run:
            print(f"\n--- SLURM script for {exp_name} ---")
            print(script_content)
            print(f"--- End script for {exp_name} ---\n")
            continue
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Submit job
            print(f"Submitting job for experiment: {exp_name}")
            result = subprocess.run(['sbatch', script_path], 
                                  capture_output=True, text=True, check=True)
            
            job_id = result.stdout.strip().split()[-1]
            submitted_jobs.append((exp_name, job_id))
            print(f"  Job ID: {job_id}")
            
            if args.sequential:
                print(f"Waiting for job {job_id} to complete...")
                subprocess.run(['squeue', '--job', job_id, '--format=%i %T %M', 
                              '--noheader'], check=False)
        
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for {exp_name}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        
        finally:
            # Clean up temporary script
            os.unlink(script_path)
    
    if not args.dry_run and submitted_jobs:
        print(f"\nSubmitted {len(submitted_jobs)} jobs:")
        for exp_name, job_id in submitted_jobs:
            print(f"  {exp_name}: {job_id}")
        
        print(f"\nTo monitor jobs, use:")
        print(f"  squeue -u $USER")
        print(f"  squeue -j {','.join([job_id for _, job_id in submitted_jobs])}")


if __name__ == "__main__":
    main()
