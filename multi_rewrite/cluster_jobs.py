r"""Tools to submit multiple jobs to the cluster.
"""
# Copyright 2019-2020 jhauschild, MIT License
# I maintain this file at https://github.com/jhauschild/cluster_jobs

from __future__ import print_function  # (for compatibility with python 2)

import pickle
import subprocess
import sys
import os
import re
import warnings
import importlib



# TODO: include saving & collecting the data?!?
class Task:
    """Abstract base class for a task ("simulation") to be run with a given set of parameters."""

    def run(self, parameters):
        """Run the task once for a given set of `parameters`."""
        print("This is an execution of a (dummy-)task with the following parameters:")
        print(parameters)


class CommandCall(Task):
    """Call a Command with given arguments."""
    def __init__(self, comm):
        self.filename = filename

    def run(self, parameters):
        cmd = ['bash', self.filename] + parameters
        print("call", cmd)
        res = subprocess.call(cmd)
        if res > 0:
            raise ValueError("Error while running command " + ' '.join(cmd))


class PythonFunctionCall(Task):
    """Task calling a specific python function of a given module."""
    def __init__(self, module, function):
        self.module = module
        self.function = function

    def run(self, parameters):
        mod = importlib.import_module(self.module)
        fct = mod
        for subpath in self.function.split('.'):
            fct = getattr(fct, subpath)
        fct(**parameters)


class TaskArray:
    """A series of tasks to be run."""

    TASK_TYPES = {
        'Dummy': Task,
        'CommandCall': CommandCall,
        'PythonFunctionCall': PythonFunctionCall
    }

    def __init__(self, task, task_parameters, **kwargs):
        if isinstance(task, dict):
            task_args = task.copy()
            type_ = task_args.pop('type')
            TaskClass = self.TASK_TYPES[type_]
            task = TaskClass(**task_args)
        self.task = task
        self.task_parameters = task_parameters

    @property
    def len(self):
        return len(self.task_parameters)

    def run_local(self, task_ids=None, parallel=False):
        """Run the tasks for the specified task_ids locally."""
        if task_ids is None:
            task_ids = range(self.len)
        if parallel:
            raise NotImplementedError("TODO")
        for taks_id in task_ids:
            self.run_task(taks_id)

    def run_task(self, task_id):
        parameters = self.task_parameters[task_id]
        self.task.run(parameters)

    def missing_output(config, key='output_filename'):
        """Return task ids for which there is no output file."""
        result = []
        for i, parameters in enumerate(self.task_parameters):
            output_filename = parameters[key]
            if not os.path.exists(output_filename):
                result.append(i)
        return result


class ScheduledTaskArray(TaskArray):

    script_templates_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "templates")

    def __init__(self,
                 jobname,
                 script_template="bash.sh",
                 config_filename_template="{jobname}.config.pkl",
                 script_filename_template="{jobname}_run.sh",
                 **kwargs):
        self.jobname = jobname
        self.script_template = script_template
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        self = cls(**config)
        return self

    @classmethod
    def from_config_file(cls, filename):
        config = read_config_file(filename)
        return cls.from_config(config)

    def write_config_file(self, filename):
        """Write the config to file `filename`."""
        if self.len == 0:
            raise ValueError("No task parameters scheduled")
        config = self.__dict__
        write_config_file(filename, config)

    def unique_filenames(self):
        """Find a unique filenames for the config and the script by adjusting the `jobname`."""
        filename_templates = [self.config_filename_template, self.script_filename_template]
        jobname = self.jobname
        check_jobnames = [jobname] + [jobname + '_{i:02}'.format(i=i) for i in range(100)]
        for jobname in check_jobnames:
            formated_filenames = [fn.format(jobname=jobname) for fn in filename_templates]
            if not any([os.path.exists(fn) for fn in formated_filenames]):
                break
        else:  # no break
            raise ValueError("Can't find unique filenames for config. Clean up!")
        self.jobname = jobname
        return formated_filenames


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

    def run_local(self, task_ids=None, parallel=False):
        """Run the tasks for the specified task_ids locally."""
        config_fn, script_fn = self.unique_filenames()

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

    def read_script_template(self, filename):
        directory = self.script_template_directory
        filename = os.path.join(directory, filename)
        with open(filename, 'r') as f:
            text = f.read()
        return text


class SGEJobArray(ScheduledTaskArray):

    def submit(config, sge_template_filename='sge_template.txt'):
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


class SlurmJobArray(TaskArray):
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
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            config = pickle.load(f)
    elif filename.endswith('.yml'):
        import yaml
        with open(filename, 'r') as f:
            config = yaml.save_load(f)
    else:
        raise ValueError("Don't recognise filetype of config file " + repr(filename))
    if not isinstance(config, dict):
        raise TypeError("expected config to be dict, got: " + str(type(config)))
    return config


def write_config_file(filename, config):
    if filename.endswith('.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(config, f, protocol=2)  # use protocol 2 if you need Python-2 support
    else:
        raise ValueError("Don't recognise filetype of config filename " + repr(filename))




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




def time_str_to_seconds(time):
    """Convert a time intervall specified as a string ``dd:hh:mm:ss'`` into seconds.

    Accepts both '-' and ':' as separators."""
    intervals = [1, 60, 60*60, 60*60*24]
    return sum(iv*int(t) for iv, t in zip(intervals, reversed(time.replace('-', ':').split(':'))))


def main_run(args):
    task_array = ScheduledTaskArray.from_config_file(args.configfile)
    task_array.run_local(args.taskid, args.parallel)


def main():
    """Parse command line arguments and perform the specified task."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Print steps taken.")
    subparsers = parser.add_subparsers(dest="command")

    parser_run = subparsers.add_parser("run", help="run a given set of tasks",
                                       description="run one or multiple tasks")
    parser_run.add_argument("--parallel", action="store_true",
                            help="if multiple tasks are given, run them in parallel.")
    parser_run.add_argument("configfile",
                            help="job configuration, e.g. 'my_jobname_02.config.pkl'")
    parser_run.add_argument("taskid", type=int, nargs="+", help="select the tasks to be run")

    parser_submit = subparsers.add_parser("submit", help="submit a task array to the cluster")
    parser_submit.add_argument("jobconfig", nargs=1,
                               help="job configuration, e.g. 'my_jobname_02.config.pkl'")
    parser_show = subparsers.add_parser("show", help="pretty-print the configuration")
    args = parser.parse_args()
    if args.verbose:
        print(args)  # TODO debug
    if args.command == "run":
        main_run(args)



if __name__ == "__main__":
    main()
