#!/bin/bash

#SBATCH --chdir ./                      # output files realtive to the directory from where we submit jobs
#SBATCH --job-name trialrun             # jobname
#SBATCH --output ./%x.%N.%j.out         # this is where the output goes. %x=Job name, %j=Jobd id, %N=node.
#SBATCH --get-user-env                  # this is needed
#SBATCH --clusters=kcs_nim              # selects the nim cluster
#SBATCH --partition=kcs_nim_batch       # ...
#SBATCH --reservation=kcs_nim_users     # ...
#SBATCH --export=NONE                   # this is needed
#SBATCH --nodes=1-1                     # select a full node (64 cores). 
#                               !!! NOTE: On the NIM-cluster, you can only choose FULL Nodes, so please submit 
#                                         enough jobs in a shared memory system (e.g. parameter sweep) to really 
#                                         make use of all 64 cores and not block them unnecessarily!!!
#SBATCH --cpus-per-task=64              # use all cpus of the node
#SBATCH --mail-type=FAIL                # you will receive a mail should your job fail. You can also choose NONE in case you don't want to be emailed
#SBATCH --mail-user=max.muster@tum.de   # you have to enter an email address here
#SBATCH --time=2-23:50:00               # enter a maximum runtime for the job. (format: days-hours:minutes:seconds)
#                                         Maximum runtime on nim cluster: 3-00:00:00  (``scontrol show partition --clusters=kcs_nim``)

source /etc/profile.d/modules.sh        # load the modules system of the LRZ

module load python/3.6_intel            # load the module intelpython.
# you can check out all available modules (e.g. python/2.7_intel if you use python2) 
#with the command `module avail` in the command line

export MKL_NUM_THREADS=64               # number of cores per node, total for all the tasks below!
export MKL_DYNAMIC=FALSE                # important: use hyperthreading and not just the number of physical cores.

echo "Execute  job on host $HOSTNAME at $(date)"
NUMBER_TASKS=16                         # how many task do you want to submit on this node?
for ((TASKID = 1; TASKID <= $NUMBER_TASKS; TASKID += 1))
do
    OUTPUTFILE="$SLURM_JOB_NAME.$TASKID.$SLURM_JOB_ID.out"   # separate output file for each task
    python ./simulation.py $TASKID &> $OUTPUTFILE &          # the actual tasks; $TASKID is an input parameters;
    echo "started task $TASKID, writing to $OUTPUTFILE"
done
wait  # ensures that the job is not terminated unless all tasks have been completed, except for exceeding the runtime
echo "finished job at $(date)"
