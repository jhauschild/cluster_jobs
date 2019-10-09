# Utilities/examples to submit jobs to our computing cluster

The folder 'simple_sge_tum_ph/' cotains a simple working example how to run a python simulation on the computing cluster of the TU Munich physics department, which is using the Sun Grid Engine (SGE) for job submission (via the `qsub` command). 
The folder 'simple_slurm_lrz_nim/' contains a simple working example how to run a python simulation on the computing cluster for the NIM cluster of the LRZ, which is using SLURM for the jub submission. 
Further explanations can be found in our group wiki at https://wiki.tum.de/display/tuphtfk/Cluster+utilisation 
(password protected, since it gives details about our cluster...).

The 'multi/' folder contains the file `jobs_tum.py` to simplify submission of many (similar) simulations using Python. 
Explanations are in the module doc string of `jobs_tum.py`. 
It provides the functions `submit_sge(...)` for SGE, `submit_slurm(...)` for SLURM, 
as well as `run_local(...)` for debugging on a local machine.
