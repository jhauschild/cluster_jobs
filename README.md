# Utilities/examples to submit jobs to our computing cluster

The folder 'simple_sge_tum_ph/' cotains a simple working example how to run a python simulation on the computing cluster of the TU Munich physics department, which is using the Sun Grid Engine (SGE) for job submission (via the `qsub` command). 
The folder 'simple_slurm_lrz_nim/' contains a simple working example how to run a python simulation on the computing cluster for the NIM cluster of the LRZ, which is using SLURM for the jub submission. 
Further explanations can be found in our group wiki at https://wiki.tum.de/display/tuphtfk/Cluster+utilisation 
(password protected, since it gives details about our cluster...).

The 'multi/' folder contains the file `cluster_jobs.py` to simplify submission of many (similar) simulations using Python. 
Explanations are in the module doc string of `cluster_jobs.py`. 
It provides the functions `submit_sge(...)` for SGE, `submit_slurm(...)` for SLURM, 
as well as `run_local(...)` for debugging on a local machine.

The `multi_rewrite/` folder contains an object-oriented rewrite of `multi/cluster_jobs.py` that should make the code a bit clearer.

The `multi_yaml/` is the newest version and contains another variant that works nicely with human-readable yaml files. This is particularly usefull to quickly inspect submitted details and have all parameters ready, and plays nicely with the simulation framework of [TeNPy](https://github.com/tenpy/tenpy).

**If you're new** to submitting jobs on the cluster, I'd recommend looking into the `simple_sge_tum_ph/` folder first to get a sense of how the cluster works, and then into the `multi_yaml/` rewrite for production runs. In the latter, you can find a detailed `README.md` with an example how to submit jobs.
