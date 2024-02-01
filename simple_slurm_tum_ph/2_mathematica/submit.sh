#!/bin/bash

# hardware requirements
#SBATCH --time=00:10:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
# note: you need to read out the $SLURM_CPUS_PER_TASK in the mathematica script to make use of the parallelization!
#SBATCH --cpus-per-task=4                  # use multi-threading with 4 cpu threads (= 2 physical cores + hyperthreading)
#SBATCH --mem=5G                           # request this amount of memory for each task

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
# #SBATCH --constraint "jammy&intel"       # select Ubuntu version and/or cpu. See `scontrol show config | grep Feature`

# to select a sepcific mathematica version, uncomment:
export PATH="/mount/packs/Mathematica13.2/:$PATH"

echo "starting job on $(hostname) at $(date) with $SLURM_CPUS_PER_TASK cores"
# example matematica script
wolfram -script ./simulation.wsl
echo "finished job at $(date)"
