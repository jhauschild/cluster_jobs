#!/bin/bash

# hardware requirements
#SBATCH --time=00:10:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --cpus-per-task=4                  # use multi-threading with 4 cpu threads (= 2 physical cores * hyperthreading)
#SBATCH --mem=5G                           # request this amount of memory (total per node)

#SBATCH --partition=cpu                    # optional, cpu is default. needed for gpu/classes. See `sinfo` for options
#SBATCH --qos=debug                        # Submit debug job for quick test. See `sacctmgr show qos` for options

# some further useful options, uncomment as needed/desired
#SBATCH --job-name MyJob                   # descriptive name shown in queue and used for output files
#SBATCH --output %x.%j.out                 # this is where the (text) output goes. %x=Job name, %j=Jobd id, %N=node.
# #SBATCH --error  %x.%j.err               # uncomment if you want separate stdout and stderr
# #SBATCH --mail-type=ALL                  # uncomment to ask for email notification.
# #SBATCH --mail-user=invalid@example.com  # email to send to. Defaults to your personal ga12abc@mytum.de address
# #SBATCH --get-user-env                   # If active, copy environment variables from submission to the the job
# #SBATCH --chdir ./                       # change to the specified directory
# #SBATCH --constraint "Ubuntu22.04&intel" # select Ubuntu version and cpu family. See `scontrol show config | grep Feature`

set -e  # abort the whole script if one command fails

# see `man sbatch` for further possible environment variables you can use
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE  # number of CPUs per node, total for all the tasks below.
# NOTE: for some applications, like BLAS/LAPACK matrix multiplication/diagonalization
# it is actually better to ignore hyperthreading, i.e. only use the number of physical cores you get:
# export OMP_NUM_THREAD=$(( $SLURM_CPUS_ON_NODE / 2 ))

# if needed, you can set/adjust PATH,PYTHONPATH etc to include other libraries
# source /mount/packs/ml-2023/bin/activate
# export PYTHONPATH="$HOME/MyLibrary"

echo "starting job on $(hostname) at $(date) with $SLURM_CPUS_ON_NODE cores"
# example call of your simulation. 
python ./simulation.py 3 1.5
# This specific example takes desired runtime [minutes] and memory [GB] as command line args.
# This allows you to check what happens e.g. if you go beyond those limits - the cluster should abort your job in that case.
# For your actual simulation, you can have abitrary, different parameters here.
# NOTE: If you have many jobs with similar parameter sets, take a look at the 1_task_array example and the "advanced" multi_yaml folder.
echo "finished job at $(date)"
