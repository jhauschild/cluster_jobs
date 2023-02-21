# Using `cluster_jobs.py` to submit jobs from yaml files

When you have to run some numerical simulations that take too long to run on a single computer,
you want to submit them to the cluster (provided by the university).
You divide your problem into smaller simulation tasks,
each of which can run on a single computer node (or a few, if your task is really heavy).
Often, you want to submit a whole array of tasks, where you change only a few parameters (e.g.
some coupling strenght in your model, or the MPS bond dimension of a DMRG run).
This file helps to submit such collections of tasks to computing clusters in a convenient way.
It can work with SGE (Sun grid engine) cluster or SLURM clusters in a unified way (and you can easily add support for more clusters, if needed.)


Let us walk through an example.
Say you have the python module `simulation.py` (an example is provided) with a function
`run_simulation` that takes certain keyword arguments.
You can look at the example, which sleeps for a few seconds and then writes data (that it might have computed) to the `output_filename` given as argument.

Say you want to run on the cluster:
```python
for a in [100, 500, 1000]:
    for c in [0.5, 1.0]:
        kwargs = {
            'a': a, 
            'b': 2*np.pi,
            'sub_params': {'c: c, d: 2}
        }
        # ensure different output filename per simulation:
        kwargs['output_filename'] = f"output_{a!s}_c_{c!s}.pkl" 
        # run following line on independent nodes in the cluster
        simulation.run_simulation(**kwargs)
```
However, you also need to specify the setup to the cluster - how much time would you estimate for each time, how much RAM do you need? The cluster needs to know that
to correctly schedule the job.
Therefore, you also have to specify a `job_config` dictionary with the required information: what cluster system do you use, what resources do you need, which parameters do you want to change?
Take a look at the `example_submit.yml`. It's a nested set of parameters in human-readable YAML format (convenient to quickly edit!).
The top entry `job_config` specifies requirements to the cluster and can be read by `cluster_jobs.py` to directly setup and submit job scripts to the cluster as follows:
```bash
cluster_jobs.py submit example_submit.yml
```
In the given example, it actually just runs the specified jobs sequentially, since `job_config.class = JobConfig`.
Try running it. You should see that it runs 6 tasks, and get a bunch of output files:

- `MyJob.config.yml` is the config of the job you just submitted, so you can easily look at what you wanted to run here.
- `MyJob.run.*` is the job file that would be submitted to the cluster with `qsub MyJob.run.sge` or `sbatch MyJob.run.slurm`. When running locally, `MyJob.run.sh` is just a simple bash script.
- Further, you get the expected `output_a_*_c_*.pkl` files, and 
- one `MyJob.task_*.out` with standard-out for each task that was run. This is where you find error messages.

Adjusting the `job_config.jobname` changes the prefix of all those files (`MyJob` in the example above). If you re-submit jobs with the same `jobname`, it appends numbers `_00, _01, _02,...` to the jobname.

To sumit to SGE or SLURM, simply change the `job_config.class` to `SGEJob` or `SlurmJob`, and make sure to specify hardware `requirements_slurms` or `requirements_sge`, respectively.

You can also specify the `job_config.script_template` to any file in the `cluster_templates/` folder. This is convenient if you need to set up custom environments
(e.g. to set PATH or PYTHONPATH environment variables) - for example, the `tenpy_v0.10_tum.sge` would set it up to use the share TeNPy installation on the TUM SGE cluster.

As an alternative to the YAML format, you can also submit jobs with a python script like `example_submit.py`.
Personally, though, I found the YAML file format much more convenient to quickly change things in a readable fashion.


## Useful directory setup
While you can save everything just somewhere in your home directory on your personal laptop, clusters of have a file system quota in $HOME, which allows you to save only a limited amount of data.
For example here at TUM Physics, output files from jobs should be written somewhere within `/space`. (Read your cluster's documentation to learn where to save stuff!) 
The `cluster_jobs.py` does not have to be in the same directory as the output files. Rather, any python source files (if you have any) and the `cluster_templates/*` folder should be in the same directory as your `cluster_jobs.py`.
A usefull setup is thus as follows:

### Setup
- Setup a folder (or even better, a git repository) with all you config and source code in your home directory, say in `~/projects/my_fancy_model/`.
  Copy the `cluster_jobs.py`, the `cluster_templates/*` folder and your simulation scripts (e.g. `simulation.py`) there.
- It can be useful to copy the `project_activate` script there (and rename it to just `activate`, if you don't like long names).
  Inside this script, setup any environment variables like `$PATH, PYTHONPATH,...` that you want to use whenever you use this project - e.g. if you want to setup a custom python environment adjust, set PYTHONPATH to point to your custom TeNPy installation or similar stuff.
  If that script adjusts the PATH, you can easily call scripts like `cluster_jobs.py` or your custom post-processing plotting scripts.
  On linux, you can acchieve that by 1) having a shebang line like `#!python` on top of the script, and making the script executable with `chmod u+x my_script.py`.
- Make subfolders for projects (using one source-code repository of what you want to do) and "attempts" within the project (trying out this and that within the project), e.g, `/space/your_username/projects/my_fancy_model/phase_diagram`, `.../quench`, `.../quench_second_attempt`, `.../quench_final` and finally `.../quench_final_FINAL` (Admit it - you do that as well, don't you?).
- It can be quite useful to create a link with `ln -s /space/your_username/projects/my_fancy_model data` from your projects home to ease the navigation - then you never need to leave $HOME despite your data actually being in `/space`.
- Adjust the submit yaml file to your needs such that you only need to change a few lines to actually submit jobs.
- Setup scripts for post-processing, e.g. plotting.

### Usage in practice
- `cd ~/projects/my_fancy_model/` to your projects folder, , and `. project_activate` (or just `. activate` if you renamed it) to adjust the PATH/PYTHONPATH etc.
- Update/fix/adjust code as needed (and git commit).
- `mkdir data/new_attempt` and `cd data/new_attempt` to the data (sub)folders.
- `cp $PROJECT_DIR/submit.yml .` and edit it as needed
- Finally call `cluster_jobs.py submit submit.yml`, check that jobs run and wait for them to finish.
- Look at your data and post-process your results.
