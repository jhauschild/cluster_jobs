#!/bin/bash

# hardware requirements
#SBATCH --time=00:10:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --nodes=2                          # on this number of nodes
#SBATCH --ntasks-per-node=3                # request this many MPI tasks per node
#SBATCH --cpus-per-task=4                  # each task using (OMP) multi-threading with this many cpu threads (= 2 physical cores * hyperthreading)
#SBATCH --mem-per-task=100M                # request this amount of memory for each MPI task

#SBATCH --partition=cpu_mpi_intel          # or cpu_mpi_amd
#SBATCH --qos=normal                       # Submit normal job. See `sacctmgr show qos` for options

# some further useful options, uncomment as needed/desired
#SBATCH --job-name MyJob                   # descriptive name shown in queue and used for output files
#SBATCH --output %x.%j.%t.out                 # this is where the (text) output goes. %x=Job name, %j=Jobd id, %N=node, %t=mpi rank
# #SBATCH --error  %x.%j.err               # uncomment if you want separate stdout and stderr
# #SBATCH --mail-type=ALL                  # uncomment to ask for email notification.
# #SBATCH --mail-user=invalid@example.com  # email to send to. Defaults to your personal ga12abc@mytum.de address
# #SBATCH --get-user-env                   # If active, copy environment variables from submission to the the job
# #SBATCH --chdir ./                       # change to the specified directory
# #SBATCH --constraint "Ubuntu24.04&intel" # select Ubuntu version and cpu family. See `scontrol show config | grep Feature`

set -e  # abort the whole script if one command fails

source /etc/os-release
source mpi_${UBUNTU_CODENAME}/bin/activate  # activate local python env. 
# create this env with 
#     python -m venv mpi
#     source mpi/bin/activate
#     pip install mpi4py

# see `man sbatch` for further possible environment variables you can use

# TODO check if SLURM_CPUS_PER_TASK is different from SLURM_CPUS_ON_NODE for this case!!!
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # number of CPUs per node, total for all the tasks below.

# if needed, you can set/adjust PATH,PYTHONPATH etc to include other libraries
# source /mount/packs/ml-2023/bin/activate
# export PYTHONPATH="$HOME/MyLibrary"

echo "starting job on $(hostname) at $(date) with $SLURM_CPUS_PER_TASK cores"
# example call of your simulation. 
srun --mpi=pmix python ./simulation.py 3 0.1 # This specific example takes desired runtime [minutes] and memory [GB] as command line args.
# This allows you to check what happens e.g. if you go beyond those limits - the cluster should abort your job in that case.
# For your actual simulation, you can have abitrary, different parameters here.
# if you have many jobs with similar parameter sets, take a look at the 1_task_array example and the "advanced" multi_yaml folder.
echo "finished job at $(date)"
