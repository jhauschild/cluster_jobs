#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -m ae # mail on 'a'bort and/or 'e'xit
{requirements}

set -e  # abort whole script if any command fails

export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS={cores_per_task:d}

# if needed, you can set PYTHONPATH here to include other libraries, e.g.
# export PYTHONPATH="$HOME/MyLibrary"

echo "Running task {task_id} of {config_file} on $HOSTNAME at $(date)"
python {cluster_jobs_py} run {config_file} {task_id}
echo "finished at $(date)"
