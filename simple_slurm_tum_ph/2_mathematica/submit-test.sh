#!/bin/bash

# submit from the supplied mathematica notebook!

# hardware requirements
#SBATCH --time=0:30:00                     # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --cpus-per-task=2                  # use multi-threading with 4 cpu threads (= 2 physical cores + hyperthreading)
#SBATCH --mem=5G                           # request this amount of memory (total per node)

#SBATCH --partition=cpu                    # optional, cpu is default. needed for gpu/classes. See `sinfo` for options
#SBATCH --qos=debug                        # Submit debug job for quick test. (max 30min) See `sacctmgr show qos` for options

# some further useful options, uncomment as needed/desired
#SBATCH --job-name SlurmMathTest           # descriptive name shown in queue and used for output files

# The following two lines are changed compared to non-array submission,
# in addition to the simlation_array.py taking the argument $SLURM_ARRAY_TASK_ID
#SBATCH --output %x.%A-%a.out              # this is where the (text) output goes. %x=Job name, %a=Jobd id, %N=node.
# #SBATCH --array=0-5                        # range that $SLURM_ARRAY_TASK_ID should take - supplied on the command line!
# you can also provide individual values as --array=0,2,5-10

# #SBATCH --error  %x.%j.err               # uncomment if you want separate stdout and stderr
# #SBATCH --mail-type=ALL                  # uncomment to ask for email notification.
# #SBATCH --mail-user=invalid@example.com  # email to send to. Defaults to your personal ga12abc@mytum.de address
# #SBATCH --get-user-env                   # If active, copy environment variables from submission to the the job
#SBATCH --chdir ./                       # change to the specified directory

#set -e  # abort the whole script if one command fails

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # number of CPUs per node, total for all the tasks below.
# see `man sbatch` for further possible environment variables you can use

echo "Starting array task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date) with $SLURM_CPUS_PER_TASK cores"
math -run < worker-test.m
echo "finished job at $(date)"
