r"""Tools to submit multiple jobs to the cluster.

A single "simulation" (e.g., DMRG) is defined by a simulation python file (say `simulation.py`)
containing a function (say `run_simulation`) wich takes a dictionary `kwargs` as arguments.
Often, we need to call the same function multiple times with only slightly different parameters,
e.g., to check convergence with a certain parameter (like the bond dimension of DMRG),
or to tune some model parameters (like tuning through a quantum phase transition in DMRG).
The basic idea of this module is that we can generate a single `config` for the job array,
which defines which function in which file should be called how often with which parameters.
The `config` is simply a dictionary with the following entries:

============ ============================================================= ======================
keyword      value                                                         example
============ ============================================================= ======================
jobname      Short descriptive name for the job. Used for filenames of     'TestJob'
             the config and jobscript and appears in output of `qstat`.
------------ ------------------------------------------------------------- ----------------------
sim_file     The file containg the simulation function to be run.          'simulation.py'
             (It is recommended to use an absolute path for this.)
------------ ------------------------------------------------------------- ----------------------
sim_fct      Name of the function in `sim_file` which should be called.    'run_simulation'
------------ ------------------------------------------------------------- ----------------------
params       List of dictionaries. For each dictionary `kwargs` in this    [{'a': 10, 'b': 1.1},
             list we perform one simulation, i.e. one job on the cluster.   {'a': 20, 'b': 1.2},
                                                                            {'a': 20, 'b': 1.1}]
------------ ------------------------------------------------------------- ----------------------
require      Dictionary of hardware requirements on the cluster.           {'cpu': '24:00:00',
             See table below for details.                                   'Nslots': 4}
------------ ------------------------------------------------------------- ----------------------
email        Optional: If given, send an email if the job gets aborted     user@example.com
             by the cluster.
------------ ------------------------------------------------------------- ----------------------
mail_on_exit Optional: if an `email` was given, also send a mail if the    False
             job finishes without interruption.
------------ ------------------------------------------------------------- ----------------------

The example would correspond to running 3 simulations::

    run_simulation(a=10, b=1.1)
    run_simulation(a=20, b=1.2)
    run_simulation(a=20, b=1.1)

Each simulation might be run on a different machine and gets the require hardware
as specified in ``config['require']``. You can use the following requirements:

============ ======= ======== ========================================================
name         type    default  description
============ ======= ======== ========================================================
Nslots       int     4        Number of threads/cpu cores to use for each simulation.
------------ ------- -------- --------------------------------------------------------
cpu          str     0:55:00  (Wall clock) run time in hours:minutes:seconds.
                              Note that if your job is not finished by the specified
                              time, it gets killed!
------------ ------- -------- --------------------------------------------------------
mem          str     2G       Memory requirement per cpu, i.e., you require a total
                              memory `Nslots * mem` for the simulaiton.
                              '1G' = 1 Gigabyte == 1024 Megabyte = '1024M'.
                              (Not used for SLURM right now.)
------------ ------- -------- --------------------------------------------------------
filesize     str     4G       Maximum size of (all) output file(s).
                              (Not used for SLURM right now.)
------------ ------- -------- --------------------------------------------------------
queue        str              Optional: specify the queue, e.g. 'cond-mat'.
                              (Not used for SLURM right now.)
============ ======= ======== ========================================================

Giving such a config dictionary to the function :func:`submit_sge` will submit a job array to our
cluster.
The file `example_submit.py` creates an example config and submits it to the cluster.

The function :func:`run_local` has the same call structure as :func:`submit_sge`,
but instead of submitting the jobs to the cluster, it just runs the simulations locally,
i.e. on the machine where you are working, one after another.
This is especially usefull for debugging and to check that your simulation
(input/output) works as expected before you burn a lot of CPU hours on the cluster.


How things work together
------------------------

Our cluster supports job arrays calling the same job script (*.sge) many times on different
machines, while just changing a *single* argument: the ``$SGE_TASK_ID`` environment variable.
It starts counting at 1, so for a given job of the job array, we execute the funciont specified in
the config with the ``kwargs = config['params'][SGE_TASK_ID-1]``.

The function :func:`submit_sge` writes the config to a file (called '{jobname}.config.pkl')
and generates and submits a jobscript (called '{jobname}.sge') to the cluster.
If the corresponding files already exist, we append '_##' to the jobname.
where `i` is an increasing number if you submit multiple jobs in the same directory),
The cluster executes this script on suitable hosts (=computers in the cluster fullfilling
the hardware requirements) with various ``$SGE_TASK_ID = 1, 2, ..., len(config['params'])``.
The job script (based on the file ``sge_template.txt`` in the same folder as `simulation.py`)
will call the `simulation.py` with two command line arguments,
namely the filename of the config file and the $SGE_TASK_ID.

Hence, the `simulation.py` needs to check its command line arguments to read the config file
and execute the desired function `run_simulation` with the correct `kwargs`.
This can conveniently done using the function :func:`run_simulation_commandline`;
simply copy & paste the following lines at the end of `simulation.py`::

    if __name__ == "__main__":
        import cluster_jobs
        cluster_jobs.run_simulation_commandline(globals())

In addition, each job executed on the cluster writes stderr and stdout (i.e. what you
`print(...)` in the simulation) to the files '{jobname}.sge.e####.#'
and '{jobname}.sge.o####.#', where ###.# is the job id assigned by the cluster.


Example
-------

First of all: don't write simulation data to $HOME, as it has a 5GB quota limit!

With the setup of this module, the python scripts for the simulation do not have to be in the
same folder as your output data: we use relative paths for the output files
(i.e., we specify only the filename, not the full path including folders).
Therefore, you can keep the python codes (example_submit.py, cluster_jobs.py, simulation.py,
sge_template.txt) in your $HOME directory, say in ~/jobs_example::

    ga12abc@homer:~/jobs_example> ls
    example_resubmit.py  example_submit.py  cluster_jobs.py  sge_template.txt  simulation.py

You can then create another folder to store your data, say /space/$USER/jobs_example::

    ga12abc@homer:~/jobs_example> mkdir -p /space/$USER/jobs_example
    ga12abc@homer:~/jobs_example> cd /space/$USER/jobs_example
    ga12abc@homer:/space/ga12abc/jobs_example>

From there, you can execute the python script "example_submit.py" to create and submit the *.sge script:

    ga12abc@homer:/space/ga12abc/jobs_example> python ~/jobs_example/example_submit.py
    ga12abc@homer:/space/ga12abc/jobs_example> ls
    MyJob_00.config.pkl  MyJob_00.sge
    ga12abc@homer:/space/ga12abc/jobs_example> qstat
    script and config file::
    job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
    -----------------------------------------------------------------------------------------------------------------
    1080341 0.00000 MyJob_00.s ga12abc      qw    03/21/2018 17:50:20                                    4 1-6:1

As you can see, submitting the job created the files MyJob_00.config.pkl and MyJob_00.sge.
The job is waiting to start as indicated by the `qw` in the `state` column of the output of qstat.
When it has run successfully, it will disappear from `qstat`, and the output files appear:

    ga12abc@homer:/space/ga12abc/jobs_example> ls
    MyJob_00.config.pkl      MyJob_00.sge.o1080341.1   MyJob_00.sge.pe1080341.3  MyJob_00.sge.po1080341.5
    MyJob_00.sge             MyJob_00.sge.o1080341.2   MyJob_00.sge.pe1080341.4  MyJob_00.sge.po1080341.6
    MyJob_00.sge.e1080341.1  MyJob_00.sge.o1080341.3   MyJob_00.sge.pe1080341.5  result_a_1000_b_0.50.pkl
    MyJob_00.sge.e1080341.2  MyJob_00.sge.o1080341.4   MyJob_00.sge.pe1080341.6  result_a_1000_b_1.00.pkl
    MyJob_00.sge.e1080341.3  MyJob_00.sge.o1080341.5   MyJob_00.sge.po1080341.1  result_a_100_b_0.50.pkl
    MyJob_00.sge.e1080341.4  MyJob_00.sge.o1080341.6   MyJob_00.sge.po1080341.2  result_a_100_b_1.00.pkl
    MyJob_00.sge.e1080341.5  MyJob_00.sge.pe1080341.1  MyJob_00.sge.po1080341.3  result_a_500_b_0.50.pkl
    MyJob_00.sge.e1080341.6  MyJob_00.sge.pe1080341.2  MyJob_00.sge.po1080341.4  result_a_500_b_1.00.pkl

The *.sge.o* files contain the output of the program,
the *.sge.e* files contain the stderr, i.e., any error messages.
The *.sge.po* and *.sge.pe* files appear only for jobs running on multiple cores (Nslots > 1) and
never contained anything for me, you can simply ignore/remove them
(they are suppose to hold stdout/stderr for parallel jobs).


Using a library (like TeNPy) and compiling
------------------------------------------

If you need an additional library in you $PYTHONPATH, you should adjust the `sge_template.txt`.
For example, simply adding the following line will allow python to import files from the specified
directory:

    export PYTHONPATH="$HOME/MyLibrary"



If you need to compile the library, things get a bit more complicated.
I provide *compiled* copies of both the old and new TeNPy, to be shared amongst our group.
These versions are used by sge_template_tenpy.txt and sge_template_prev_tenpy.txt.

If the library is still under heavy development and you change stuff while jobs are running,
the cleanest and most straigh-forward solution is to copy and compile the library in the sge
script to a temporary directory, i.e., once for each job which you run.
An example for that is given in the file `sge_template_compile.txt`.
One drawback of this apprach is that aborted jobs can leave 'ghost' copies of the library behind,
which you need to clean up by hand.

Another possibility is to compile the library once for the whole cluster. This works well as long
as the cluster is set up sufficiently uniform regarding its hardware, e.g., that all nodes run
64-bit CPUs. For the gcc compiler, you should then use the option ``-march=x86-64``.

Another approach that works surprisingly well on many cluster is to use the
$HOSTNAME (i.e., name of the machine running a simulation) to determine a directory where you
stored a pre-compiled version of the desired library.
Since this is not needed in most cases, I don't provide an example;


Using the NIM cluster with SLURM
--------------------------------
Our group has access to the NIM cluster at LRZ.
Read our group wiki on how to gain access and login there.
The idea is that this file can be used in the exact same way as on the other cluster,
except that you call the function ``submit_slurm(...)`` instead of ``submit_sge(...)``.

Unfortunately, it is only possible to reserve full nodes at the NIM cluster right now,
which makes things a little bit more complicated. Since there are only 14 nodes, at most 14 people
can get a job running at the same time.
Each node has (with hyperthreading) 64 cores and a total of 370GB memory (i.e. 5.8GB per core),
so you have quite some computational power with a single node. Make sure that your job can use it!


"""
# Copyright 2019-2020 jhauschild, MIT License
# I maintain this file at https://github.com/jhauschild/cluster_jobs in multi/

from __future__ import print_function  # (for compatibility with python 2)

import pickle
import subprocess
import sys
import os
import re
import warnings

__all__ = ['submit_sge', 'submit_slurm', 'run_local', 'run_simulation_commandline',
           'get_filenames_config_jobscript', 'read_textfile', 'create_sge_script',
           'create_slurm_script', 'read_config_file', 'write_config_file', 'output_missing']


def submit_sge(config, sge_template_filename='sge_template.txt'):
    """Submit a job (array) to the SGE

    The SGE (Sun Grid Engine) is the program managing the cluster at the physics department of
    TU Munich.

    Parameters
    ----------
    config : dict
        Job configuration. See module doc-string for details.
    sge_template_filename : str
        Filename for the template of the *.sge script.
        We look for this file in the folder of `config['sim_file']`.
    """
    check_config(config)
    config_filename, jobscript_filename = get_filenames_config_jobscript(
        config, "{jobname!s}.config.pkl", "{jobname!s}.sge")
    write_config_file(config_filename, config)
    sge_template = read_textfile(config, sge_template_filename)
    create_sge_script(jobscript_filename, sge_template, config_filename, config)
    cmd_submit = ['qsub', jobscript_filename]
    print(' '.join(cmd_submit))
    try:  # submit the job
        subprocess.call(cmd_submit)
    except OSError as e:
        raise OSError("Failed to submit the job. Are you on a submit host?") from e


def submit_slurm(config, slurm_template_filename='slurm_template.txt'):
    """Submit a job (array) to SLURM.

    SLURM is the cluster management of the LRZ in Munich.

    Parameters
    ----------
    config : dict
        Job configuration. See module doc-string for details.
    slurm_template_filename : str
        Filename for the template of the jobscript.
        We look for this file in the folder of `config['sim_file']`.
    """
    check_config(config)
    config_filename, jobscript_filename = get_filenames_config_jobscript(
        config, "{jobname!s}.config.pkl", "{jobname!s}.sh")
    write_config_file(config_filename, config)
    slurm_template = read_textfile(config, slurm_template_filename)
    create_slurm_script(jobscript_filename, slurm_template, config_filename, config)
    cmd_submit = ['sbatch', jobscript_filename]
    print(' '.join(cmd_submit))
    try:  # submit the job
        subprocess.call(cmd_submit)
    except OSError as e:
        raise OSError("Failed to submit the job. Are you on a submit host?") from e


def run_local(config, sge_template_filename='sge_template.txt'):
    """Run the job(s) specified by `config` locally.

    Parameters
    ----------
    config : dict
        Job configuration. See module doc-string for details.
    sge_template : str | None
        If not None, create and execute an sge script  `submit_sge` does
        We look for this file in the folder of `config['sim_file']`.
    """
    check_config(config)
    config_filename, jobscript_filename = get_filenames_config_jobscript(
        config, "{jobname!s}.config.pkl", "{jobname!s}.sge")
    write_config_file(config_filename, config)
    if sge_template_filename is not None:
        # create sge script like submit_sge()
        sge_template = read_textfile(config, sge_template_filename)
        create_sge_script(jobscript_filename, sge_template, config_filename, config)

    # run jobs
    for job_id in range(1, len(config['params']) + 1):  # (start counting at 1)
        if sge_template_filename is not None:
            os.environ['SGE_TASK_ID'] = str(job_id)
            cmd = ['/usr/bin/env', 'bash', jobscript_filename]
        else:
            cmd = ['python', config['sim_file'], config_filename, str(job_id)]
        print("=" * 80)
        print("running $> ", ' '.join(cmd))
        print("=" * 80)
        res = subprocess.call(cmd)
        if res > 0:
            print("Error while running $> " + ' '.join(cmd))
            sys.exit(1)
    # done


def check_config(config):
    """Check that the config has the expected form of a dictionary with expected keys.

    Parameters
    ----------
    config : dict
        Job configuration. See module doc-string for details.

    Raises
    ------
    UserWarning : if the config contains unexpected keys, suggesting typos.
    ValueError : if something else is wrong with the config.
    """
    config_keys = ['jobname', 'sim_file', 'sim_fct', 'params', 'require', 'email', 'mail_on_exit']
    requ_keys = ['Nslots', 'cpu', 'mem', 'filesize', 'queue']
    for key in config:
        if key not in config_keys:
            warnings.warn("unexpected key (typo?) in `config`: {0!r}".format(key))
    for key in config['require']:
        if key not in requ_keys:
            warnings.warn("unexpected key (typo?) in `config['require']`: {0!r}".format(key))
    for key in config_keys[:-2]:
        if key not in config:
            raise ValueError("missing required key {0!r} in `config`".format(key))
    if 'email' in config:
        email = str(config['email'])
        if not re.match("[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("email '{}' doesn't look like an email".format(email))
    if len(config['params']) == 0:
        raise ValueError("Empty parameters")


def run_simulation_commandline(module_functions):
    """Run simulation specified by the command line arguments.

    Expects the command line arguments `CONFIG_FILE` and `JOB_ID`.
    Reads the config file to obtian the function name of the simluation and
    keyword arguments of the specified job_id

    Parameters
    ----------
    module_functions : dictionary
        Dictionary containing the global functions of `sim_file` specified in the config.
        Use ``globals()`` if you call this function from another script.
    """
    # parse command line arguments
    try:
        config_file, job_id = sys.argv[1:]
        job_id = int(job_id)
    except ValueError:
        raise ValueError("Invalid command line arguments. Expected arguments: config_file job_id")
    config = read_config_file(config_file)
    sim_func = module_functions[config['sim_fct']]
    kwargs = config['params'][job_id-1]
    sim_func(**kwargs)


def get_filenames_config_jobscript(config, conf_fn, jobscript_fn):
    """Find an `i` to get filenames for config and sgescript which don't exist yet.

    Parameters
    ----------
    config : dict
        Job configuration. See module doc-string for details.
    conf_fn, jobscript_fn : str
        Templates (i.e. strings to be formated with the keyword `jobname`)
        for the filenames of the config and jobscript.

    Returns
    -------
    config_filename , jobscirpt_filename : str
        Filename for the config file and job script, accoring to the templates.
        If the files already exist, we append ``_00, _01, _02, ...`` to the jobname, until
        we find a free slot for the jobname.
    """
    jobname = config['jobname']
    files = set(os.listdir('.'))  # files in the current directory
    check_jobnames = [jobname] + [jobname + '_{i:02}'.format(i=i) for i in range(100)]
    for jobname in check_jobnames:
        config_filename = conf_fn.format(jobname=jobname)
        jobscript_filename = jobscript_fn.format(jobname=jobname)
        if config_filename not in files and jobscript_filename not in files:
            break
    else:  # no break
        raise ValueError("Can't find unique filenames for config and sgescript. Clean up!")
    config['jobname'] = jobname  # put modified jobname back in config.
    return config_filename, jobscript_filename


def read_textfile(config, filename='sge_template.txt'):
    """Read a text file from the same folder as `config['sim_file']`.

    Parameters
    ----------
    config : dict
        Job configuration. See module doc-string for details.
    template_filename : str
        Filename of the file to be read, e.g., for the template of the jobscript.
        We look for this file in the folder of `config['sim_file']`.

    Returns
    -------
    jobscript_template : str
        Content of the file, e.g., the template for the jobscript.
    """
    sim_file_dir = os.path.dirname(os.path.abspath(config['sim_file']))
    filename = os.path.join(sim_file_dir, filename)
    # read template script
    with open(filename, 'r') as f:
        text = f.read()
    return text


def create_sge_script(jobscript_filename, sge_template, config_filename, config):
    """Create a submission script for the Sun Grid Engine(SGE).

    This function uses ``sge_template.format(**replacements)`` to replace hardware
    requirements and necessary filenames in the `sge_template`,
    and writes the formatted script to `jobscript_filename`.

    The `sge_template` can/should contain the following replacements:

    ============= =====================================================================
    name
    ============= =====================================================================
    cpu           Hardware requirements as specified in the module doc-string.
    mem
    filesize
    Nslots        If `Nslots` > 1, add the corresponding settings to `more_options`.
    ------------- ---------------------------------------------------------------------
    more_options  Used to insert additional/optional SGE options, e.g. for `queue`,
                    if `Nslots` > 1 needs to be specified to the SGE, or
                    for the optional line with the email.
                    Should be somewhere at the beginning of the file
                    (before actual bash commands).
    ------------- ---------------------------------------------------------------------
    sim_file      Filename of the simulation as defined in the `config`.
    ------------- ---------------------------------------------------------------------
    config_file   Filename of the `config`.
    ------------- ---------------------------------------------------------------------
    job_id        Set to '$SGE_TASK_ID' if ``len(config['params']) > 1``;
                  otherwise defaults to '1'.
    ------------- ---------------------------------------------------------------------
    email         The user's email (if given).
    ============= =====================================================================


    Parameters
    ----------
    jobscript_filename : str
        Filename where to write the sge script.
    sge_template : str
        String to be formatted with the replacements.
    config_filename : str
        Filename where to the config can be found.
    config : dict
        Job configuration. See module doc-string for details.
    """
    replacements = dict(jobname=config['jobname'],
                        sim_file=config['sim_file'],
                        sim_fct=config['sim_fct'],
                        config_file=config_filename,
                        **config['require'])
    N_tasks = len(config['params'])
    # set default replacements
    replacements.setdefault('cpu', '0:55:00')
    replacements.setdefault('mem', '2G')
    replacements.setdefault('filesize', '4G')
    replacements.setdefault('Nslots', 4)
    cpu_seconds = time_str_to_seconds(replacements['cpu'])
    replacements.setdefault('cpu_seconds', cpu_seconds)
    more_options = replacements.get('more_options', "")

    # adjust SGE script options
    if replacements['Nslots'] > 1:
        # specify for parallel environment, give the number of threads/cpus
        more_options += "#$ -pe smp {Nslots:d}\n#$ -R y\n".format(**replacements)
    if N_tasks > 1:
        # use a job-array with different $SGE_TASK_ID
        more_options += "#$ -t 1-{N_tasks:d}\n".format(N_tasks=N_tasks)
        replacements.setdefault('job_id', "$SGE_TASK_ID")
    elif N_tasks < 1:
        raise ValueError("Got no parameters in job config!")
    else:
        replacements.setdefault('job_id', "1")
    if 'queue' in replacements:
        more_options += "#$ -q " + str(replacements['queue']) + "\n"
    if 'email' in config:
        email = str(config['email'])
        replacements['email'] = email
        more_options += "#$ -M " + email + "\n"
        mail_option = 'ae' if config.get('mail_on_exit', False) else 'a'
        more_options += "#$ -m {0} # a=MAIL_AT_ABORT, e=MAIL_AT_EXIT\n".format(mail_option)
    replacements['more_options'] = more_options

    # format template
    sge_script = sge_template.format(**replacements)
    # write the script to file
    with open(jobscript_filename, 'w') as f:
        f.write(sge_script)


def create_slurm_script(jobscript_filename,
                        slurm_template,
                        config_filename,
                        config,
                        N_cores_per_node=64):
    """Create a submission script for the Sun Grid Engine(SGE).

    This function uses ``slurm_template.format(**replacements)`` to replace hardware
    requirements and necessary filenames in the `slurm_template`,
    and writes the formatted script to `jobscript_filename`.

    The `slurm_template` can/should contain the following replacements:

    ============= =====================================================================
    name
    ============= =====================================================================
    cpu           Hardware requirements as specified in the module doc-string.
    mem
    Nslots
    ------------- ---------------------------------------------------------------------
    more_options  Used to insert additional/optional SGE options, e.g.
                  for a job-array if more than one task is to be submitted.
                  Should be somewhere at the beginning of the file
                  (before actual bash commands).
    ------------- ---------------------------------------------------------------------
    sim_file      Filename of the simulation as defined in the `config`.
    ------------- ---------------------------------------------------------------------
    config_file   Filename of the `config`.
    ------------- ---------------------------------------------------------------------
    job_id        Defaults to '1' if just one node was requested; otherwise
                  set to '$SLURM_ARRAY_TASK_ID' (starting to count at 1).
    ------------- ---------------------------------------------------------------------
    task_starts   Desired limits for the $TASKID variable in the `slurm_template`.
    task_stops    Each of them is a string of whitespace separated integers,
                  one integer for each `job_id`.
    ============= =====================================================================


    Parameters
    ----------
    jobscript_filename : str
        Filename where to write the slurm script.
    slurm_template : str
        String to be formatted with the replacements.
    config_filename : str
        Filename where to the config can be found.
    config : dict
        Job configuration. See module doc-string for details.
    N_cores_per_node : int
        Number of cores per node. In our setup, the node is blocked completely by a single job,
        having that number of cores available. Hence, we put an appropriate number of tasks in a
        single job.
    """
    replacements = dict(jobname=config['jobname'],
                        sim_file=config['sim_file'],
                        sim_fct=config['sim_fct'],
                        config_file=config_filename,
                        **config['require'])
    N_tasks = len(config['params'])
    # set default replacements
    replacements.setdefault('cpu', '0:55:00')
    N_slots = replacements.setdefault('Nslots', 4)
    cpu_seconds = time_str_to_seconds(replacements['cpu'])
    replacements.setdefault('cpu_seconds', cpu_seconds)
    more_options = replacements.get('more_options', "")

    if 'mem' in replacements:
        more_options += "#SBATCH --mem-per-cpu={mem!s}\n".format(**replacements)

    if N_tasks < 1:
        raise ValueError("Got no parameters in job config!")
    if N_tasks == 1:
        if N_slots < N_cores_per_node:
            print("WARNING: We're always blocking a full node with {cores:d} cores. "
                  "Can your one task really use that?".format(cores=N_cores_per_node))
        replacements.setdefault('job_id', "1")
        replacements.setdefault('task_starts', "1")
        replacements.setdefault('task_stops', "1")
    else:
        assert N_slots >= 1
        asked_N_cores = N_slots * N_tasks
        N_nodes = asked_N_cores // N_cores_per_node
        if asked_N_cores < N_cores_per_node or \
                asked_N_cores % N_cores_per_node > N_cores_per_node / 2.:
            N_nodes += 1
            msg = ("Asking for {N_tasks:d} tasks of each {N_slots:d} cores, "
                   "rounding up to {N_nodes:d} node(s).")
        elif asked_N_cores % N_cores_per_node == 0:
            msg = ("Asking for {N_tasks:d} tasks of each {N_slots:d} cores, "
                   "fitting into {N_nodes:d} node(s).")
        else:
            msg = ("Asking for {N_tasks:d} tasks of each {N_slots:d} cores, "
                   "rounding down to {N_nodes:d} node(s).")
        print(msg.format(N_tasks=N_tasks, N_nodes=N_nodes, N_slots=N_slots))

        if N_nodes > 1:
            more_options += "#SBATCH --array=1-{N_nodes:d}\n".format(N_nodes=N_nodes)
            replacements.setdefault('job_id', "$SLURM_ARRAY_TASK_ID")
        else:
            replacements.setdefault('job_id', "1")

        # distribute tasks over nodes
        tasks_per_node = [0] * N_nodes  # how many tasks for each node?
        for i in range(N_tasks):
            tasks_per_node[i % N_nodes] += 1
        task_starts = []
        task_stops = []
        start = 1
        for node in range(N_nodes):
            task_starts.append(str(start))
            task_stops.append(str(start + tasks_per_node[node] - 1))
            start += tasks_per_node[node]
        replacements.setdefault('task_starts', ' '.join(task_starts))
        replacements.setdefault('task_stops', " ".join(task_stops))

    if 'email' in config:
        email = config['email']
        replacements['email'] = config['email']
        replacements['mailtype'] = "FAIL"
        if config.get('mail_on_exit', False):
            replacements['mailtype'] = "FAIL,END"
    else:
        replacements['email'] = ""
        replacements['mailtype'] = "NONE"
    replacements['more_options'] = more_options

    # format template
    slurm_script = slurm_template.format(**replacements)
    # write the script to file
    with open(jobscript_filename, 'w') as f:
        f.write(slurm_script)


def read_config_file(filename):
    """Read config file `filename`."""
    if filename[-4:] != '.pkl':
        raise ValueError("Unknown file format for config file: " + str(filename))
    with open(filename, 'rb') as f:
        config = pickle.load(f)
    return config


def write_config_file(filename, config):
    """Write the config to file `filename`."""
    if len(config['params']) == 0:
        raise ValueError("No configurations scheduled")
    with open(filename, 'wb') as f:
        pickle.dump(config, f, protocol=2)


def output_missing(config, filename_kw='output_filename'):
    """Return indices of `config['params']` for which the output is missing.

    Parameters
    ----------
    config : dict
        Job configuration. See module doc-string for details.
    filename_kw : str
        Keyword of the `kwargs` given to the simulation function `config['sim_fct']` to specify
        the simulation output filename.
    """
    result = []
    for i, kwargs in enumerate(config['params']):
        output_filename = kwargs[filename_kw]
        if not os.path.exists(output_filename):
            result.append(i)
    return result


def time_str_to_seconds(time):
    """Convert a time intervall specified as a string ``dd:hh:mm:ss'`` into seconds.

    Accepts both ',' and ':' as separators."""
    intervals = [1, 60, 60*60, 60*60*24]
    return sum(iv*int(t) for iv, t in zip(intervals, reversed(time.replace('-', ':').split(':'))))


if __name__ == "__main__":
    # if this file is called from the command line,
    # (pretty) print the content of a config file
    try:
        from pprint import pprint
    except ImportError:
        pprint = print
    for file in sys.argv[1:]:
        print("="*80)
        print(file)
        config = read_config_file(file)
        pprint(config)
