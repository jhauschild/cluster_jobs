#!/bin/bash
#SBATCH --mail-type=fail
{requirements}

set -e  # abort whole script if any command fails

# === prepare the environement as necessary ===
# module load python/3.7
# conda activate tenpy

export OMP_DYNAMIC=False
export OMP_NUM_THREADS={cores_per_task:d}

echo "Running task {task_id} specified in {config_file} on $HOSTNAME at $(date)"
python {cluster_jobs_py} run {config_file} {task_id}
echo "finished at $(date)"
