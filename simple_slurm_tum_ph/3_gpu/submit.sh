#!/bin/bash

#SBATCH --job-name MyJob                   # descriptive name show in queue and used for output files
#SBATCH --time=00:10:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --cpus-per-task=4                  # use multi-threading with 4 cpu threads (= 2 physical cores + hyperthreading)
#SBATCH --mem=5G                           # request this amount of memory for each task

#SBATCH --qos=debug                        # Submit debug job for quick test. See `sacctmgr show qos` for options
 
# you need the following settings for students jobs
#SBATCH --gpus=1                           # request the gpu
#SBATCH --partition=gpu                    # on the gpu partition
#SBATCH --account=gpu_manual               # with a non-default account (limited access!)
#SBATCH --reservation=gpu                  # using the cpu/ram set aside


#SBATCH --output %x.%j.out                 # this is where the (text) output goes. %x=Job name, %j=Jobd id, %N=node.

# some further useful options, uncomment if desired
# #SBATCH --error  %x.%j.err               # uncomment if you want separate stdout and stderr
# #SBATCH --mail-type=ALL                  # uncomment to ask for email notification.
# #SBATCH --mail-user=invalid@example.com  # email to send to. Defaults to your personal ga12abc@mytum.de address
# #SBATCH --get-user-env                   # If active, copy environment variables from submission to the the job
# #SBATCH --chdir ./                       # change to the specified directory
# #SBATCH --constraint "jammy&intel"       # select Ubuntu version and/or cpu. See `scontrol show config | grep Feature`

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # number of CPUs per node, total for all the tasks below.
# see `man sbatch` for further possible environment variables you can use

# if needed, you can set/adjust PATH,PYTHONPATH etc to include other libraries
# source /mount/packs/ml-2023/bin/activate
# export PYTHONPATH="$HOME/MyLibrary"

echo "starting gpu job on $(hostname) at $(date) with $SLURM_CPUS_PER_TASK cores"
# some more debug info
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi  # check that CPU is available
# example running for 3 minutes using 1.5 GB
python ./simulation.py  3 1.5
echo "finished job at $(date)"
