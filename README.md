## Utilities/examples to submit jobs to our computing cluster

The folder 'simple_slurm_tum_ph/' cotains a simple working example how to run a python simulation on the computing cluster of the TU Munich physics department, which is using [SLURM](https://slurm.schedmd.com) for job submission (via the `sbatch` command). 
The folder 'simple_slurm_lrz_nim/' contains a simple working example how to run a python simulation on the computing cluster for the NIM cluster of the LRZ, which is also using SLURM, but nodes are not shared, you can only reserve full nodes.
Further explanations can be found at the wiki page https://wiki.tum.de/display/nat/SLURM+queuing+system (You need a TUM login to read it.)

## Advanced submission with `cluster_jobs.py`

The `multi*` folders contain an enhanced setup based around a python script `cluster_jobs.py`, that can aid you (a lot) in submitting multiple jobs at once changing only a few parameters.
The basic idea is that the `cluster_jobs.py` can generate (and even directly submit) the job script that you would usually submit with `qsub my_job.sh`. However, it sets up a so-called task array on the cluster, which runs the same job many times for slightly different parameters (that you specified before).

The different `multi*` folders are essentially different versions of this same idea developed over time, with increasing leavel of sophistication.
The `multi/` folder is the first (legacy) version I wrote, `multi_rewrite` was a rewrite in an object-oriented coding style, and `multi_yaml` is based on that, but extended to support the [yaml file format](https://en.wikipedia.org/wiki/YAML) for submission.


The `multi/cluster_jobs.py` has an extensive doc string in the beginning with example usage. The `multi_rewrite/cluster_jobs.py` can essentially be called the same way. The [multi_yaml](https://github.com/jhauschild/cluster_jobs/tree/main/multi_yaml) has an extensive README with explanations, walking you through an example and hinting at a usefull setup.

**If you're new to submitting jobs** on the cluster, I'd recommend looking first into the `simple_slurm_tum_ph/` folder first to get a sense of how the cluster works, and then into the `multi_yaml/` folder.

Note: the `cluster_jobs.py` it not specific to SLURM, it can also handle SGE and easily adjusted to the setup at other university clusters.
