"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import copy

config = {
    'jobname': 'MyJob',
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'simulation',
        'function': 'run_simulation'
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements': {
        # 'memory': '4G',
        'time': '0:30:00',  # d-hh:mm:ss
        # 'nodes': 2,
        # 'cores_per_job',    # d-hh:mm:ss
    },
}

for a in [100, 500, 1000]:
    for b in [0.5, 1.]:
        kwargs = {'a': a, 'b': b}
        kwargs['output_filename'] = 'result_a_{a:d}_b_{b:.2f}.pkl'.format(a=a, b=b)
        config['task_parameters'].append(copy.deepcopy(kwargs))

# cluster_jobs.TaskArray(**config).run_local()
cluster_jobs.ScheduledTaskArray(**config).run_local()
# cluster_jobs.SlurmJobArray(**config).submit()
# cluster_jobs.SGEJobArray(**config).submit()
