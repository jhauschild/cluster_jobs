#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --chdir=./
#SBATCH --output=./{jobname}.%A_%3a.out  # %J=jobid.step, %N=node.
#
# To support getting emails, adjust the following two lines and remove the `# `,i.e. make them start with `#SBATCH `
# #SBATCH --mail-type fail  # or `fail,end`, but it's not recommended
# #SBATCH --mail-user your.email@tum.de  # adjust...
# NOTE: use ONLY YOUR UNIVERSITY EMAIL, DON'T USE/FORWARD EMAIL to other email providers like gmail.com!
# You can get a lot of emails from the cluster, and other email providers then sometimes mark the whole university as sending spam.
# This might results in your professor not being able to write emails to his friends anymore...
{requirements}

set -e  # abort whole script if any command fails

# === prepare the environement as necessary ===
# module load python/3.7
# conda activate tenpy
{environment_setup}

# use SLURM_CPUS_PER_TASK, if not set default to SLURM_CPUS_ON_NODE
USE_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-${{SLURM_CPUS_ON_NODE}}}}
if [ -z "$USE_NUM_THREADS" ]
then
	USE_NUM_THREADS="$(nproc --all)"
	echo "WARNING: SLURM_CPUS_ON_NODE not set! Using all cores on machine, NTHREADS=$USE_NUM_THREADS"
fi
export OMP_NUM_THREADS=$USE_NUM_THREADS  # number of CPUs per node, total for all the tasks below.
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=$USE_NUM_THREADS  # number of CPUs per node, total for all the tasks below.
export NUMBA_NUM_THREADS=$USE_NUM_THREADS

echo "Running task {task_id} specified in {config_file} on $HOSTNAME at $(date) with $USE_NUM_THREADS threads"
python {cluster_jobs_module} run {config_file} {task_id}
# if you want to redirect output to file, you can append the following to the line above:
#     &> "{jobname}.task_{task_id}.out"
echo "finished at $(date)"
