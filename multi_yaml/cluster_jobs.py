#!/usr/bin/env python3
"""Tools to submit multiple jobs to the cluster.

"""
# Copyright 2021 jhauschild, MIT License
# I maintain this file at https://github.com/jhauschild/cluster_jobs in multi_yaml/

# requires Python >= 3.5

import pickle
import subprocess
import os
import sys
import warnings
import importlib
from pprint import pprint, PrettyPrinter
import itertools
from collections.abc import Mapping
from copy import deepcopy
import sys

from io import StringIO

# --------------------------------------
# Classes for Tasks
# --------------------------------------


class Task(object):
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
        print("call", ' '.join(cmd))
        res = subprocess.call(cmd)
        if res > 0:
            raise ValueError("Error while running command " + ' '.join(cmd))

    def __repr__(self):
        return "CommandCall({0!r})".format(' '.join(self.command))


class PythonFunctionCall(Task):
    """Task calling a specific python function of a given module."""
    def __init__(self, module, function, extra_imports=None):
        self.module = module
        self.function = function
        self.extra_imports = extra_imports

    def run(self, parameters):
        if self.extra_imports is not None:
            for module in self.extra_imports:
                print('import ', module)
                importlib.import_module(module)
        mod = importlib.import_module(self.module)
        fct = mod
        for subpath in self.function.split('.'):
            fct = getattr(fct, subpath)
        fct(**parameters)

    def __repr__(self):
        return "PythonFunctionCall({0!r}, {1!r}, {2!r})".format(self.module,
                                                                self.function,
                                                                self.extra_imports)


class TaskArray(object):
    """The same task to be run multiple times with different parameters.

    We define the `task_id` to start from 1 instead of 0,
    since cluster engines like SGE and SLURM support that better.
    To avoid having to subtract/add 1 to the task_id each time, we just include a `None` entry
    at the very beginning of `self.task_parameters`, such that
    ``self.task_parameters[1]`` is the first defined set of parameters for `task_id` 1.
    """

    task_types = {
        'Task': Task,
        'CommandCall': CommandCall,
        'PythonFunctionCall': PythonFunctionCall
    }

    def __init__(self,
                 task,
                 task_parameters,
                 filter_task_ids=None,
                 verbose=False,
                 output_filename_keys=['output_filename'],
                 **kwargs):
        if isinstance(task, dict):
            task_args = task.copy()
            type_ = task_args.pop('type')
            TaskClass = self.task_types[type_]
            task = TaskClass(**task_args)
        self.task = task
        self.task_parameters = [None] + list(task_parameters)
        if filter_task_ids is not None:
            self.task_parameters = [None] + [self.task_parameters[i] for i in filter_task_ids]
        self.verbose = verbose
        self.output_filename_keys = output_filename_keys

    @property
    def N_tasks(self):
        return len(self.task_parameters) - 1

    @property
    def task_ids(self):
        return range(1, len(self.task_parameters))

    def run_local(self, task_ids=None, parallel=1, only_missing=False):
        """Run the tasks for the specified task_ids locally."""
        if task_ids is None:
            task_ids = self.task_ids
        if only_missing:
            task_ids = [i for i in self.task_ids if self.is_missing_output(i)]
        if parallel == 1:
            for task_id in task_ids:
                self.run_task(task_id)
        elif parallel > 1:
            from multiprocessing import Pool
            pool = Pool(parallel)
            pool.map(self.run_task, task_ids)
        else:
            raise ValueError("parallel={0!r} doesn't make sense!".format(parallel))

    def run_task(self, task_id):
        if task_id == 0:
            raise ValueError("`task_id` starts counting at 1!")
        parameters = self.task_parameters[task_id]
        self.task.run(parameters)

    def task_ids_missing_output(self, keys=None):
        """Return task_ids for which there is no output file."""
        if keys is None:
            keys = self.output_filename_keys
        result = []
        for task_id in self.task_ids:
            if task_id == 0:
                continue
            if self.is_missing_output(task_id, keys):
                result.append(task_id)
        return result

    def is_missing_output(self, task_id, keys=None):
        if keys is None:
            keys = self.output_filename_keys
        for key in keys:
            parameters = self.task_parameters[task_id]
            output_filename = parameters[key]
            if not os.path.exists(output_filename):
                return True
        return False

    def __repr__(self):
        res = StringIO()
        res.write(self.__class__.__name__ + "\n")
        pp = PrettyPrinter(stream=res)
        config = self.__dict__.copy()
        del config['task_parameters']
        pp.pprint(config)
        return res.getvalue()[:-1]

# --------------------------------------
# Classes for cluster jobs
# that setup scripts
# --------------------------------------

class JobConfig(TaskArray):

    source_dir = os.path.dirname(os.path.abspath(__file__))
    script_templates_dir = os.path.join(source_dir, "cluster_templates")

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
        super(JobConfig, self).__init__(**kwargs)

    @classmethod
    def expand_from_config(cls, **config):
        job_config = config['job_config'].copy()
        task_parameters = config.copy()
        if job_config.get('remove_this_section', True):
            del task_parameters['job_config']
        task_parameters = expand_parameters(task_parameters,
                                            **job_config.get('change_parameters', {}))
        job_config['task_parameters'] = task_parameters
        self = cls(**job_config)
        self.expanded_from = config
        self.config_filename_template = self.config_filename_template[:-3] + 'yml'
        return self

    def submit(self):
        """Submit the task tasks for the specified task_ids locally."""
        self.options['task_id'] = "$TASK_ID"
        script_file = self.prepare_submit()
        for task_id in self.task_ids:
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
        print("Write job config with {0:d} tasks to {1!r}".format(self.N_tasks, filename))
        if filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(self, f, protocol=2)  # use protocol 2 to still support Python 2
        elif filename.endswith('.yml'):
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(self.expanded_from, f, sort_keys=False)
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
        o['jobname'] = self.jobname
        o['config_file'] = config_file
        o.setdefault('cluster_jobs_module', __file__)
        o.setdefault('environment_setup', "")
        o.setdefault('N_tasks', self.N_tasks)
        requirements_lines = []
        for key, value in self.requirements.items():
            requirements_lines.append(self.get_requirements_line(key, value))
            key = key.replace('-', '_').replace(' ', '_')  # try to convert to identifier
            o[key] = value
        o['requirements'] = '\n'.join(requirements_lines).format(**o)
        script = script.format(**o)
        with open(script_file, 'w') as f:
            f.write(script)

    def get_requirements_line(self, key, value):
        return "#REQUIRE {key}={value}".format(key=key, value=value)


class SGEJob(JobConfig):
    def __init__(self, requirements_sge={}, **kwargs):
        self.requirements_sge = requirements_sge
        kwargs.setdefault('script_template', "sge.sh")
        kwargs.setdefault('script_filename_template', "{jobname}.sge.sh")
        super(SGEJob, self).__init__(**kwargs)

    def submit(self):
        if self.N_tasks > 1000:
            raise ValueError("Refuse to submit {0:d} tasks at once.".format(self.N_tasks))
        script_file = self.prepare_submit()
        cmd_submit = ['qsub', script_file]
        print(' '.join(cmd_submit))
        subprocess.call(cmd_submit)

    def update_requirements(self):
        o = self.options
        r = self.requirements
        r.update(self.requirements_sge)
        if self.N_tasks == 1:
            o['task_id'] = 1
        else:
            r.setdefault('t', '1-{N_tasks:d}')
            o['task_id'] = "$SGE_TASK_ID"
        if o.setdefault("cores_per_task", 4) > 1:
            r.setdefault('pe smp', "{cores_per_task:d}")
            r.setdefault('R', "y")

    def get_requirements_line(self, key, value):
        return "#$ -{key} {value}".format(key=key, value=value)


class SlurmJob(JobConfig):
    def __init__(self, requirements_slurm={}, **kwargs):
        self.requirements_slurm = requirements_slurm
        kwargs.setdefault('script_template', "slurm.sh")
        kwargs.setdefault('script_filename_template', "{jobname}.slurm.sh")
        super(SlurmJob, self).__init__(**kwargs)

    def submit(self):
        if self.N_tasks > 1000:
            raise ValueError("Refuse to submit {0:d} tasks at once.".format(self.N_tasks))
        script_file = self.prepare_submit()
        cmd_submit = ['sbatch', script_file]
        print(' '.join(cmd_submit))
        subprocess.call(cmd_submit)

    def update_requirements(self):
        o = self.options
        r = self.requirements
        r.update(self.requirements_slurm)
        if self.N_tasks == 1:
            o['task_id'] = 1
        else:
            r.setdefault('array', '1-{N_tasks:d}')
            o['task_id'] = "$SLURM_ARRAY_TASK_ID"

    def get_requirements_line(self, key, value):
        return "#SBATCH --{key}={value}".format(key=key, value=value)


job_classes = {'TaskArray': TaskArray,
               'JobConfig': JobConfig,
               'SGEJob': SGEJob,
               'SlurmJob': SlurmJob}


# --------------------------------------
# Function to parse command line args
# --------------------------------------
# get_recursive, set_recursive, and merge_recursive are as defined in `tenpy.tools.misc`;
# but copy-pasted here to avoid import errors when TeNpy is not installed


_UNSET = object()  # sentinel


def get_recursive(nested_data, recursive_key, separator=".", default=_UNSET):
    """Extract specific value from a nested data structure.

    Parameters
    ----------
    nested_data : dict of dict (-like)
        Some nested data structure supporting a dict-like interface.
    recursive_key : str
        The key(-parts) to be extracted, separated by `separator`.
        A leading `separator` is ignored.
    separator : str
        Separator for splitting `recursive_key` into subkeys.
    default :
        If not specified, the function raises a `KeyError` if the recursive_key is invalid.
        If given, return this value when any of the nested dicts does not contain the subkey.

    Returns
    -------
    entry :
        For example, ``recursive_key="some.sub.key"`` will result in extracing
        ``nested_data["some"]["sub"]["key"]``.

    See also
    --------
    set_recursive : same for changing/setting a value.
    """
    if recursive_key.startswith(separator):
        recursive_key = recursive_key[len(separator):]
    if not recursive_key:
        return nested_data  # return the original data if recursive_key is just "/"
    for subkey in recursive_key.split(separator):
        if default is not _UNSET and subkey not in nested_data:
            return default
        nested_data = nested_data[subkey]
    return nested_data


def set_recursive(nested_data, recursive_key, value, separator=".", insert_dicts=False):
    """Same as :func:`get_recursive`, but set the data entry to `value`."""
    if recursive_key.startswith(separator):
        recursive_key = recursive_key[len(separator):]
    subkeys = recursive_key.split(separator)
    for subkey in subkeys[:-1]:
        if insert_dicts and subkey not in nested_data:
            nested_data[subkey] = {}
        nested_data = nested_data[subkey]
    nested_data[subkeys[-1]] = value


def update_recursive(nested_data, update_data, separator=".", insert_dicts=True):
    """Wrapper around :func:`set_recursive` to allow updating multiple values at once.

    It simply calls :func:`set_recursive` for each ``recursive_key, value in update_data.items()``.
    """
    for k, v in update_data.items():
        set_recursive(nested_data, k, v, separator, insert_dicts)


def merge_recursive(*nested_data, conflict='error', path=None):
    """Merge nested dictionaries `nested1` and `nested2`.

    Parameters
    ----------
    *nested_data: dict of dict
        Nested dictionaries that should be merged.
    path: list of str
        Path inside the nesting for useful error message
    conflict: "error" | "first" | "last"
        How to handle conflicts: raise an error (if the values are different),
        or just give priority to the first or last `nested_data` that still has a value,
        even if they are different.

    Returns
    -------
    merged: dict of dict
        A single nested dictionary with the keys/values of the `nested_data` merged.
        Dictionary values appearing in multiple of the `nested_data` get merged recursively.
    """
    if len(nested_data) == 0:
        raise ValueError("need at least one nested_data")
    elif len(nested_data) == 1:
        return nested_data[0]
    elif len(nested_data) > 2:
        merged = nested_data[0]
        for to_merge in nested_data[1:]:
            merged = merge_recursive(merged, to_merge, conflict=conflict, path=path)
        return merged
    nested1, nested2 = nested_data
    if path is None:
        path = []
    merged = nested1.copy()
    for key, val2 in nested2.items():
        if key in merged:
            val1 = merged[key]
            if isinstance(val1, Mapping) and isinstance(val2, Mapping):
                merged[key] = merge_recursive(val1, val2,
                                              conflict=conflict,
                                              path=path + [repr(key)])
            else:
                if conflict == 'error':
                    if val1 != val2:
                        path = ':'.join(path + [repr(key)])
                        msg = '\n'.join([f"Conflict with different values at {path}; we got:",
                                         repr(val1), repr(val2)])
                        raise ValueError(msg)
                elif conflict == 'first':
                    pass
                elif conflict == 'last':
                    merged[key] = val2
        else:
            merged[key] = val2
    return merged


def expand_parameters(nested, *,
                      recursive_keys=None,
                      value_lists=None,
                      format_strs=None,
                      expansion='product',
                      output_filename=None,
                      output_filename_params_key=None,
                      separator='.',
                      ):
    if recursive_keys is None:
        return [deepcopy(nested)]
    if value_lists is None:
        value_lists = [get_recursive(nested, key, separator) for key in recursive_keys]
    if output_filename is not None:
        if output_filename_params_key is not None:
            raise ValueError("Specify either output_filename or output_filename_params_key "
                             "in job_config")
        out_fn_key = output_filename.pop('key', 'output_filename')
        output_filename.setdefault('separator', separator)
        parts = output_filename.setdefault('parts', {})
        if format_strs is None:
            format_strs = [rkey.split(separator)[-1] + '_{0!s}' for rkey in recursive_keys]
        for key, fstr in zip(recursive_keys, format_strs):
            parts[key] = fstr
    elif output_filename_params_key is None:
        if 'output_filename_params' in nested:
            output_filename_params_key = 'output_filename_params'
    if output_filename_params_key is not None:
        parts_key = separator.join([output_filename_params_key, 'parts'])
        parts = get_recursive(nested, parts_key, separator=separator, default={})
        format_strs = [rkey.split(separator)[-1] + '_{0!s}' for rkey in recursive_keys]
        for key, fstr in zip(recursive_keys, format_strs):
            parts.setdefault(key, fstr)
        set_recursive(nested, parts_key, parts, separator=separator, insert_dicts=True)

    expanded = []
    if expansion == 'product':
        iterator = itertools.product(*value_lists)
    elif expansion == 'zip':
        iterator = zip(*value_lists)
    for vals in iterator:
        new_nested = deepcopy(nested)
        for key, value in zip(recursive_keys, vals):
            set_recursive(new_nested, key, value, separator, True)
        if output_filename is not None:
            fn = output_filename_from_dict(new_nested, **output_filename)
            set_recursive(new_nested, out_fn_key, fn, separator, True)
        expanded.append(new_nested)
    return expanded


# note: this is the same as tenpy.simulations.simulation.output_filename_from_dict
def output_filename_from_dict(options,
                              parts={},
                              prefix='result',
                              suffix='.h5',
                              joint='_',
                              parts_order=None,
                              separator='.'):
    """Format a `output_filename` from parts with values from nested `options`.

    The results of a simulation are ideally fixed by the simulation class and the `options`.
    Unique filenames could be obtained by including *all* options into the filename, but this
    would be a huge overkill: it suffices if we include the options that we actually change.
    This function helps to keep the length of the output filename at a sane level
    while ensuring (hopefully) sufficient uniqueness.

    Parameters
    ----------
    options : (nested) dict
        Typically the simulation parameters, i.e., options passed to :class:`Simulation`.
    parts :: dict
        Entries map a `recursive_key` for `options` to a `format_str` used
        to format the value, i.e. we extend the filename with
        ``format_str.format(get_recursive(options, recursive_key, separator))``.
        If `format_str` is empty, no part is added to the filename.
    prefix, suffix : str
        First and last part of the filename.
    joint : str
        Individual filename parts (except the suffix) are joined by this string.
    parts_order : None | list of keys
        Optionally, an explicit order for the keys of `parts`.
        By default (None), just the keys of `parts`, i.e. the order in which they appear in the
        dictionary; before python 3.7 (where the order is not defined) alphabetically sorted.
    separator : str
        Separator for :func:`~tenpy.tools.misc.get_recursive`.

    Returns
    -------
    output_filename : str
        (Hopefully) sufficiently unique filename.

    Examples
    --------
    >>> options = {  # some simulation parameters
    ...    'algorithm_params': {
    ...         'dt': 0.01,  # ...
    ...    },
    ...    'model_params':  {
    ...         'Lx': 3,
    ...         'Ly': 4, # ...
    ...    }, # ... and many more options ...
    ... }
    >>> output_filename_from_dict(options)
    'result.h5'
    >>> output_filename_from_dict(options, suffix='.pkl')
    'result.pkl'
    >>> output_filename_from_dict(options, parts={'model_params.Ly': 'Ly_{0:d}'}, prefix='check')
    'check_Ly_4.h5'
    >>> output_filename_from_dict(options, parts={
    ...         'algorithm_params.dt': 'dt_{0:.3f}',
    ...         'model_params.Ly': 'Ly_{0:d}'})
    'result_dt_0.010_Ly_4.h5'
    >>> output_filename_from_dict(options, parts={
    ...         'algorithm_params.dt': 'dt_{0:.3f}',
    ...         ('model_params.Lx', 'model_params.Ly'): '{0:d}x{1:d}'})
    'result_dt_0.010_3x4.h5'
    >>> output_filename_from_dict(options, parts={
    ...         'algorithm_params.dt': '_dt_{0:.3f}',
    ...         'model_params.Lx': '_{0:d}',
    ...         'model_params.Ly': 'x{0:d}'}, joint='')
    'result_dt_0.010_3x4.h5'
    """
    formatted_parts = [prefix]
    if parts_order is None:
        if sys.version_info < (3, 7):
            # dictionaries are not ordered -> sort keys alphabetically
            parts_order = sorted(parts.keys(), key=lambda x: x[0] if isinstance(x, tuple) else x)
        else:
            parts_order = parts.keys()  # dictionaries are ordered, so use that order
    else:
        assert set(parts_order) == set(parts.keys())
    for recursive_key in parts_order:
        format_str = parts[recursive_key]
        if not format_str:
            continue
        if not isinstance(recursive_key, tuple):
            recursive_key = (recursive_key, )
        vals = [get_recursive(options, r_key, separator) for r_key in recursive_key]
        part = format_str.format(*vals)
        formatted_parts.append(part)
    return joint.join(formatted_parts) + suffix

# --------------------------------------
# yaml with !py_eval "..."
# --------------------------------------

try:
    import yaml
except ImportError:
    pass
else: # no ImportError
    class YamlLoaderWithPyEval(yaml.FullLoader):
        pass

    def yaml_eval_construcor(loader, node):
        """yaml constructor to support `!py_eval "... (python eval code)..."` in yaml files."""
        cmd = loader.construct_scalar(node)
        if not isinstance(cmd, str):
            raise ValueError("expect string argument to `!py_eval`")
        glob = {}
        if "np." in cmd:
            import numpy as np
            glob['np'] = np
        try:
            res = eval(cmd, glob)
        except:
            print("\nError while yaml parsing the following !py_eval command:\n", cmd, "\n")
            raise
        if isinstance(res, np.ndarray) and res.ndim == 1 and len(res) < 20:
            # try to simplify to a list of python scalars
            # such make a subsequent `yaml.dump()` much prettier
            if res.dtype.kind == 'f':
                res = [float(v) for v in res]
            elif res.dtype.kind == 'i':
                res = [int(v) for v in res]
            else:
                pass
        return res

    yaml.add_constructor("!py_eval", yaml_eval_construcor, Loader=YamlLoaderWithPyEval)

# --------------------------------------
# Function to parse command line args
# --------------------------------------

def read_config_file(filename):
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            config = pickle.load(f)
        if isinstance(config, dict):
            config = JobConfig(**config)
    elif filename.endswith('.yml'):
        import yaml
        with open(filename, 'r') as f:
            config = yaml.load(f, Loader=YamlLoaderWithPyEval)
        cls_name = config['job_config']['class']
        cls = job_classes[cls_name]
        config = cls.expand_from_config(**config)
    else:
        raise ValueError("Don't recognise filetype of config file " + repr(filename))
    return config


def time_str_to_seconds(time):
    """Convert a time intervall specified as a string like ``dd-hh:mm:ss'`` into seconds.

    Accepts both '-' and ':' as separators at all positions."""
    intervals = [1, 60, 60 * 60, 60 * 60 * 24]
    return sum(
        iv * int(t) for iv, t in zip(intervals, reversed(time.replace('-', ':').split(':'))))


def parse_commandline_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Print steps taken.")
    parser.add_argument("-m",
                        "--missing",
                        action="store_true",
                        help="only for task_ids where the output is missing")
    subparsers = parser.add_subparsers(dest="command")
    # run
    parser_run = subparsers.add_parser("run",
                                       help="run a given set of tasks",
                                       description="run one or multiple tasks")
    parser_run.add_argument("--parallel",
                            type=int,
                            default=1,
                            help="Run that many tasks in parallel.")
    parser_run.add_argument("configfile", help="job configuration, e.g. 'myjob.config.pkl'")
    parser_run.add_argument("task_id", type=int, nargs="+", help="select the tasks to be run")
    # submit
    parser_submit = subparsers.add_parser("submit", help="submit a task array to the cluster")
    parser_submit.add_argument('-c', '--conflict-merge',
                               choices=['error', 'first', 'last'],
                               default='error')
    parser_submit.add_argument("yaml_parameter_file", nargs="+",
                               help="One or multiple yaml files with the task parameters as a "
                               "single dictionary. It should have a section `job_config` "
                               "with keyword arguments for the corresponding "
                               "JobConfig/SGEJob/SlurmJob class in `cluster_jobs.py`. "
                               "The class itself is given under they key 'job_config':'class'. "
                              )
    # show
    parser_show = subparsers.add_parser("show", help="pretty-print the configuration")
    parser_show.add_argument("what", choices=["parameters", "files", "task_ids", "config"])
    parser_show.add_argument("-k", "--key", default=None,
                             help="optionally select `key` for `parameters`")
    parser_show.add_argument("configfile", help="job configuration, e.g. 'myjob.config.pkl'")
    parser_show.add_argument("task_id", type=int, nargs="*",
                             help="task id(s) for which to show stuff; default: all or --missing")
    args = parser.parse_args()
    return args


def main(args):
    if args.command == "run":
        task_array = read_config_file(args.configfile)
        task_array.run_local(args.task_id, args.parallel, args.missing)
    elif args.command == "show":
        job = read_config_file(args.configfile)
        task_ids = args.task_id
        if len(task_ids) == 0:
            task_ids = job.task_ids
        if args.missing:
            missing = job.task_ids_missing_output()
            task_ids = sorted(set(task_ids) & set(missing))
        if args.what == "files":
            for task_id in task_ids:
                for key in job.output_filename_keys:
                    fn = job.task_parameters[task_id][key]
                    if not os.path.exists(fn):
                        print(fn)
        elif args.what == "task_ids":
            print(*task_ids)
        elif args.what == "config":
            print(job)
        elif args.what == "parameters":
            for task_id in task_ids:
                if len(task_ids) > 1:
                    print("-" * 20, "task_id", task_id)
                parameters = job.task_parameters[task_id]
                if args.key is not None:
                    entry = get_recursive(parameters, args.key)
                    print(entry)
                else:
                    pprint(parameters)
        else:
            raise ValueError("unknown choice of 'what'")
    elif args.command == "submit":
        import yaml
        yaml_configs = []
        for fn in args.yaml_parameter_file:
            with open(fn, 'r') as f:
                yaml_configs.append(yaml.load(f, Loader=YamlLoaderWithPyEval))
        config = merge_recursive(*yaml_configs, conflict=args.conflict_merge)
        cls_name = config['job_config']['class']
        cls = job_classes[cls_name]
        job = cls.expand_from_config(**config)
        if args.missing:
            task_ids = job.task_ids_missing_output()
            if len(task_ids) == 0:
                print("no output files missing")
                return
            job.task_parameters = [None] + [job.task_parameters[i] for i in task_ids]
            job.expanded_from['job_config']['filter_task_ids'] = task_ids
        job.submit()
    else:
        raise ValueError("unknown command " + str(command))


if __name__ == "__main__":
    args = parse_commandline_arguments()
    main(args)
