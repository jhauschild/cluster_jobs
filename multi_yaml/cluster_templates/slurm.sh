#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output ./{jobname}.%J.out  # %J=jobid.step, %N=node.
#SBATCH --chdir=./
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

echo "Running task {task_id} specified in {config_file} on $HOSTNAME at $(date)"
python {cluster_jobs_module} run {config_file} {task_id} &> "{jobname}.task_{task_id}.out"
echo "finished at $(date)"
