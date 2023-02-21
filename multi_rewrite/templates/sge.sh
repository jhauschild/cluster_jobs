#!/bin/bash
#$ -S /bin/bash
#$ -cwd
# #$ -m ae # mail on 'a'bort and/or 'e'xit
# #$ -M your.name@university.gov  # don't use/forward to gmail.com etc
# # you can get a lot of automated emails, and other email providers might block the whole university for sending too many emails, thinking it's spam.
{requirements}

set -e  # abort whole script if any command fails

# === prepare the environement as necessary ===
# module load python/3.7
# conda activate tenpy
{environment_setup}

echo "Running task {task_id} of {config_file} on $HOSTNAME at $(date)"
python {cluster_jobs_module} run {config_file} {task_id} &> "{jobname}.task_{task_id}.out"
echo "finished at $(date)"
