#!/usr/bin/env python
"""Tools to submit multiple jobs to the cluster.


"""
# Copyright 2019-2020 jhauschild, MIT License
# I maintain this file at https://github.com/jhauschild/cluster_jobs

from __future__ import print_function  # (for compatibility with python 2)

import pickle
import subprocess
import os
import warnings
import importlib

try:
    from pprint import pprint
except:
    pprint = print


# TODO: include saving & collecting the data?!?
class Task:
    """Abstract base class for a task ("simulation") to be run with a given set of parameters."""

    def run(self, parameters):
        """Run the task once for a given set of `parameters`."""
        print("This is an execution of a (dummy-)task with the following parameters:")
        print(parameters)


class CommandCall(Task):
    """Call a Command with given arguments."""
    def __init__(self, command):
        self.command = command

    def run(self, parameters):
        cmd = [self.command] + parameters
        # TODO: escape parameters
        print("call", cmd)
        res = subprocess.call(cmd)
        if res > 0:
            raise ValueError("Error while running command " + ' '.join(cmd))

    def __repr__(self):
        return "CommandCall({0!s})".format(self.command)


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

    def __repr__(self):
        return "PythonFunctionCall({0!s}, {1!s})".format(self.module, self.function)


class TaskArray:
    """The same task to be run multiple times with different parameters."""

    task_types = {
        'Task': Task,
        'CommandCall': CommandCall,
        'PythonFunctionCall': PythonFunctionCall
    }

    def __init__(self, task, task_parameters, verbose=False, **kwargs):
        if isinstance(task, dict):
            task_args = task.copy()
            type_ = task_args.pop('type')
            TaskClass = self.task_types[type_]
            task = TaskClass(**task_args)
        self.task = task
        self.task_parameters = task_parameters
        self.verbose = verbose

    @property
    def N_tasks(self):
        return len(self.task_parameters)

    def run_local(self, task_ids=None, parallel=False):
        """Run the tasks for the specified task_ids locally."""
        if task_ids is None:
            task_ids = range(self.N_tasks)
        if parallel:
            raise NotImplementedError("TODO")
        for taks_id in task_ids:
            self.run_task(taks_id)

    def run_task(self, task_id):
        parameters = self.task_parameters[task_id]
        self.task.run(parameters)

    def missing_output(self, key='output_filename'):
        """Return task ids for which there is no output file."""
        result = []
        for i, parameters in enumerate(self.task_parameters):
            output_filename = parameters[key]
            if not os.path.exists(output_filename):
                result.append(i)
        return result


class JobConfig(TaskArray):

    source_dir = os.path.dirname(os.path.abspath(__file__))
    script_templates_dir = os.path.join(source_dir, "templates")

    def __init__(self,
                 jobname="MyJob",
                 requirements={},
                 options={},
                 script_template="run.sh",
                 config_filename_template="{jobname}.config.pkl",
                 script_filename_template="{jobname}.run.sh",
                 **kwargs):
        self.jobname = jobname
        self.requirements = requirements
        self.options = options
        self.script_template = script_template
        self.config_filename_template = config_filename_template
        self.script_filename_template = script_filename_template
        super().__init__(**kwargs)

    def submit(self):
        """Submit the task tasks for the specified task_ids locally."""
        self.options['task_id'] = "$TASK_ID"
        script_file = self.prepare_submit()
        for task_id in range(self.N_tasks):
            os.environ['TASK_ID'] = str(task_id)
            cmd = ['/usr/bin/env', 'bash', script_file]
            res = subprocess.call(cmd)
            if res > 0:
                raise ValueError("Error while running command " + ' '.join(cmd))
        # done

    def prepare_submit(self):
        config_file, script_file = self.unique_filenames()
        self.write_config_file(config_file)
        self.update_requirements()
        self.create_script(script_file, config_file)
        return script_file

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

    def write_config_file(self, filename):
        """Write the config to file `filename`."""
        if self.N_tasks == 0:
            raise ValueError("No task parameters scheduled")
        if filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(self, f, protocol=2)  # use protocol 2 to still support Python 2
        else:
            raise ValueError("Don't recognise filetype of config filename " + repr(filename))

    def read_script_template(self):
        filename = os.path.join(self.script_templates_dir, self.script_template)
        with open(filename, 'r') as f:
            text = f.read()
        return text

    def update_requirements(self):
        self.options['task_id'] = "$TASK_ID"

    def create_script(self, script_file, config_file):
        script = self.read_script_template()
        o = self.options
        o.update()
        o['config_file'] = config_file
        o.setdefault('cluster_jobs_py', __file__)
        o.setdefault('N_tasks', self.N_tasks)
        requirements = []
        for key, value in self.requirements.items():
            requirements.append(self.get_requirements_line(key, value))
            key = key.replace('-', '_').replace(' ', '_')  # try to convert to identifier
            o[key] = value
        o['requirements'] = '\n'.join(requirements).format_map(o)
        script = script.format_map(o)  # similar as script.format(**o), but allow general dict `o`
        with open(script_file, 'w') as f:
            f.write(script)

    def get_requirements_line(self, key, value):
        return "#REQUIRE {key}={value}".format(key=key, value=value)


class SGEJob(JobConfig):
    requirements_escape = "#$ -"

    def __init__(self, **kwargs):
        kwargs.setdefault('script_template', "sge.sh")
        kwargs.setdefault('script_filename_template', "{jobname}.sge.sh")
        super().__init__(**kwargs)

    def submit(self):
        script_file = self.prepare_submit()
        cmd_submit = ['qsub', script_file]
        print(' '.join(cmd_submit))
        subprocess.call(cmd_submit)

    def update_requirements(self):
        o = self.options
        r = self.requirements
        if self.N_tasks > 1:
            o['task_id'] = 0
        else:
            r.setdefault('t', '1-{N_tasks:d}')
            o['task_id'] = "$(expr $SGE_TASK_ID - 1)"
        if o.setdefault("cores_per_task", 4) > 1:
            r.setdefault('pe smp', "{cores_per_task:d}")
            r.setdefault('R', "y")

    def get_requirements_line(self, key, value):
        return "#$ -{key} {value}".format(key=key, value=value)

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
        replacements.setdefault('time', '0:55:00')
        replacements.setdefault('memory', '2G')
        replacements.setdefault('filesize', '4G')
        replacements.setdefault('Nslots', 4)
        cpu_seconds = time_str_to_seconds(replacements['cpu'])
        replacements.setdefault('cpu_seconds', cpu_seconds)
        more_options = replacements.get('more_options', "")


        # format template
        sge_script = sge_template.format(**replacements)
        # write the script to file
        with open(jobscript_filename, 'w') as f:
            f.write(sge_script)


class SlurmJob(JobConfig):
    def __init__(self, **kwargs):
        kwargs.setdefault('script_template', "slurm.sh")
        kwargs.setdefault('script_filename_template', "{jobname}.slurm.sh")
        super().__init__(**kwargs)

    def submit(self):
        script_file = self.prepare_submit()
        cmd_submit = ['sbatch', script_file]
        print(' '.join(cmd_submit))
        subprocess.call(cmd_submit)

    def update_requirements(self):
        o = self.options
        r = self.requirements
        if self.N_tasks > 1:
            o['task_id'] = 0
        else:
            r.setdefault('task', '1-{N_tasks:d}')
            o['task_id'] = "$(expr $SLURM_ARRAY_TASK_ID - 1)"
        # if o.setdefault("cores_per_task", 4) > 1:


    def get_requirements_line(self, key, value):
        return "#SBATCH --{key}={value}".format(key=key, value=value)

    def create_slurm_script(jobscript_filename,
                            slurm_template,
                            config_filename,
                            config,
                            N_cores_per_node=64):

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
    if isinstance(config, dict):
        config = JobConfig(**config)
    return config


def time_str_to_seconds(time):
    """Convert a time intervall specified as a string ``dd:hh:mm:ss'`` into seconds.

    Accepts both '-' and ':' as separators."""
    intervals = [1, 60, 60*60, 60*60*24]
    return sum(iv*int(t) for iv, t in zip(intervals, reversed(time.replace('-', ':').split(':'))))

def parse_commandline_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Print steps taken.")
    subparsers = parser.add_subparsers(dest="command")

    parser_run = subparsers.add_parser("run", help="run a given set of tasks",
                                       description="run one or multiple tasks")
    parser_run.add_argument("--parallel", action="store_true",
                            help="Run tasks in parallel.")
    parser_run.add_argument("configfile",
                            help="job configuration, e.g. 'myjob.config.pkl'")
    parser_run.add_argument("taskid", type=int, nargs="+", help="select the tasks to be run")

    parser_submit = subparsers.add_parser("submit", help="submit a task array to the cluster")
    parser_show = subparsers.add_parser("show", help="pretty-print the configuration")
    parser_show.add_argument("-m", "--missing", choices=["files", "ids"], default=None,
                               help="print the files or task ids of jobs where the output is missing.")
    parser_show.add_argument("-k", "--key", default="output_filename",
                               help="Parameter name giving the output file; default 'output_file'")
    parser_show.add_argument("configfile",
                             help="job configuration, e.g. 'myjob.config.pkl'")
    args = parser.parse_args()
    return args


def main(args):
    if args.command == "run":
        task_array = read_config_file(args.configfile)
        task_array.run_local(args.taskid, args.parallel)
    elif args.command == "show":
        job = read_config_file(args.configfile)
        if args.missing:
            missing_ids = job.missing_output(args.key)
            if args.missing == "files":
                if args.verbose:
                    print("The following output files have not been produced:")
                for task_id in missing_ids:
                    print(config['task_parameters'][task_id][args.key])
            elif args.missing == "task_ids":
                print(*missing_ids)
        else:
            pprint(config)
    else:
        raise ValueError("unknown command " + str(command))



if __name__ == "__main__":
    args = parse_commandline_arguments()
    main(args)
