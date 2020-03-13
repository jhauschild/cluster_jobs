#!/bin/bash
#$ -S /bin/bash
#$ -cwd
# hardware requirements:
#$ -l h_vmem={mem!s},h_cpu={cpu!s},h_fsize={filesize!s}
{more_options!s}

set -e   # abort whole script if any command fails

export PATH="/mount/packs/intelpython36/bin:$PATH"
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS={Nslots:d}

# if needed, you can set PYTHONPATH here to include other libraries, e.g.
# export PYTHONPATH="$HOME/MyLibrary"

echo "Execute job on host $HOSTNAME at $(date)"
echo "python {sim_file!s} {config_file!s} {job_id!s}"
python {sim_file!s} {config_file!s} {job_id!s}
echo "finished job at $(date)"
