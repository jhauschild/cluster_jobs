"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import os

config = {
    'jobname': 'MyJob',
    'email': 'YOUR-MAIL',
    'sim_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation.py'),
    'sim_fct': 'run_simulation',
    'require': {
        'mem': '4G',
        'cpu': '0:59:00',
        'Nslots': 4
    },
    'params': []  # list of dictionaries (containing the kwargs given to the function `sim_fct`)
}

for a in [100, 500, 1000]:
    for b in [0.5, 1.]:
        kwargs = {'a': a, 'b': b}
        kwargs['output_filename'] = 'result_a_{a:d}_b_{b:.2f}.pkl'.format(a=a, b=b)
        config['params'].append(kwargs.copy())

cluster_jobs.submit_sge(config)    # our linux cluster at TUM Physics department
#  cluster_jobs.submit_slurm(config)  # NIM cluster at LRZ
#  cluster_jobs.run_local(config)     # alternative to run the simulation directly
