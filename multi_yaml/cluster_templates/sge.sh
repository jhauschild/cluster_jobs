#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#
# To support getting emails, adjust the following two lines and remove the `# `,i.e. make them start with `#$ `
# #$ -m a   # mail if your jobs aborts. You can also `-m ae` to mail if the job ends, but it's not recommended.
# #$ -M your.email@tum.de  # adjust...
# NOTE: use ONLY YOUR UNIVERSITY EMAIL, DON'T USE/FORWARD EMAIL to other email providers like gmail.com!
# You can get a lot of emails from the cluster, and other email providers then sometimes mark the whole university as sending spam.
# This might results in your professor not being able to write emails to his friends anymore...
# 
{requirements}


set -e  # abort whole script if any command fails

# === prepare the environement as necessary ===
# module load python/3.7
# conda activate tenpy
{environment_setup}

export PATH="/mount/packs/intelpython36/bin:$PATH"
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=$NSLOTS   # This inserts the *same* number as specified in the line ``#$ -pe smp ...`` above
export OMP_NUM_THREADS=$NSLOTS   # This inserts the *same* number as specified in the line ``#$ -pe smp ...`` above
export NUMBA_NUM_THREADS=$NSLOTS
echo "NSLOTS=$NSLOTS"

echo "Running task {task_id} of {config_file} on $HOSTNAME at $(date)"
python {cluster_jobs_module} run {config_file} {task_id}
# if you want to redirect output to file, you can append the following to the line above:
#     &> "{jobname}.task_{task_id}.out"
echo "finished at $(date)"
