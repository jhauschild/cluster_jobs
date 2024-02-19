#!/bin/bash

# hardware requirements
#SBATCH --job-name MyJob                   # descriptive name show in queue and used for output files
#SBATCH --time=00:10:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --cpus-per-task=4                  # use multi-threading with 4 cpu threads (= 2 physical cores + hyperthreading)
#SBATCH --mem=5G                           # request this amount of memory for each task

#SBATCH --partition=cpu                    # optional, cpu is default. needed for gpu/classes. See `sinfo` for options
#SBATCH --qos=debug                        # Submit debug job for quick test. See `sacctmgr show qos` for options

# some further useful options, uncomment as needed/desired
#SBATCH --job-name MyJob                   # descriptive name shown in queue and used for output files

# The following two lines are changed compared to non-array submission,
# in addition to the simlation_array.py taking the argument $SLURM_ARRAY_TASK_ID
#SBATCH --output %x.%A-%a.out              # this is where the (text) output goes. %x=Job name, %a=Jobd id, %N=node.
#SBATCH --array=0-5                        # range that $SLURM_ARRAY_TASK_ID should take 
# you can also provide individual values as --array=0,2,5-10

# #SBATCH --error  %x.%j.err               # uncomment if you want separate stdout and stderr
# #SBATCH --mail-type=ALL                  # uncomment to ask for email notification.
# #SBATCH --mail-user=invalid@example.com  # email to send to. Defaults to your personal ga12abc@mytum.de address
# #SBATCH --get-user-env                   # If active, copy environment variables from submission to the the job
# #SBATCH --chdir ./                       # change to the specified directory
# #SBATCH --constraint "Ubuntu22.04&intel" # select Ubuntu version and cpu family. See `scontrol show config | grep Feature`

set -e  # abort the whole script if one command fails

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # number of CPUs per node, total for all the tasks below.
# see `man sbatch` for further possible environment variables you can use

# if needed, you can set/adjust PATH,PYTHONPATH etc to include other libraries
# source /mount/packs/ml-2023/bin/activate
# export PYTHONPATH="$HOME/MyLibrary"

echo "starting job on $(hostname) at $(date) with $SLURM_CPUS_PER_TASK cores"
# example array simulation
python ./simulation_array.py  $SLURM_ARRAY_TASK_ID
echo "finished job at $(date)"
