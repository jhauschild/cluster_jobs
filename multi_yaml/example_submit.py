"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import copy
import numpy as np  # only needed if you use np below

config = {
    'jobname': 'MyJob',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'simulation',
        'function': 'run_simulation'
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements_slurm': {  # passed on to SLURM
        # 'memory': '4G',
        'time': '0:30:00',  # d-hh:mm:ss
        'nodes': 1,  # number of nodes
        # 'mail-user': "no@example.com",
    },
    #  'requirements_sge': {  # for SGE
    #      'l': 'h_cpu=0:30:00,h_rss=4G',
    #      'q': 'queue',
    #      # 'M': "no@example.com"
    #  },
    'options': {  # further replacements for the job script; used to determine extra requirements
        # 'mail': 'no@example.com',
        'cores_per_task': 4,
    }
}

for a in [100, 500, 1000]:
    for c in [0.5, 1.]:
        kwargs = {
            'a': a,
            'b': 2*np.pi,
            'sub_params': {'c': c, 'd': 2}
        }
        # ensure different output filename per simulation:
        kwargs['output_filename'] = f'result_a_{a:d}_c_{c:.2f}.pkl'
        config['task_parameters'].append(copy.deepcopy(kwargs))

# cluster_jobs.TaskArray(**config).run_local(task_ids=[2, 3], parallel=2) # run selected tasks
cluster_jobs.JobConfig(**config).submit()  # run all tasks locally by creating a bash job script
# cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
# cluster_jobs.SGEJob(**config).submit()  # submit to SGE
