"""Example simulation file, containing a function that (usually) runs for a very long time...

To run your own simulations, simply edit this file by your liking.
After adjusting the hardware requirements in the `submit.sh`, you can
submit a job to our cluster with `sbatch submit.sh`.

Check the status of the cluster with `squeue --clusters=kcs_nim`.
"""

from __future__ import print_function
import numpy as np
import pickle
import time
import sys


def run_simulation(**kwargs):
    """Example simulation.

    You could imagine this function to run a costly simulation (e.g. DMRG).

    Parameters to this function are arbitrary kewyword-arguments,
    which you can read out from the dictionary.
    """
    print("executing run_simulation() in file", __file__)
    print("got the dictionary kwargs =", kwargs)

    # HERE is where you would usually run your simulation (e.g. DMRG).
    # simulate some heavy calculations:
    for i in range(30):
        print("step ", i, flush=True)  # (remove `flush=True` for Python 2)
        # the flush=True makes the output appear immediately
        time.sleep(5)

    results = {'kwargs': kwargs, 'example_data': np.random.random((2, 2))}

    output_filename = kwargs['output_filename']
    print("save results to ", output_filename)
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)


def another_simulation(output_filename, a, b):
    """Another example simulation."""
    print("executing another_simulation() in file", __file__)
    print("got arguments a={a!s},b={b!s}".format(a=a, b=b))
    # HERE is where you would usually run your simulation (e.g. some time evolution).
    time.sleep(30)  # simulate simulation: wait for 30 second
    results = {'example_data': np.random.random((3, 3))}

    print("save results to ", output_filename)
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # define set of parameters to be chosen
    parameters = [(a, b) for a in range(8) for b in [0.8, 1.2]]
    # !!! `parameters` has to have exactly 16 entries
    # choose the set of parameters from the task number (`task_id` in submit.sh)
    a, b = parameters[int(sys.argv[1]) - 1]  # note: `task_id` starts counting at 1!
    filename = "output_simple_a_{a:d}_b_{b:f}.pkl".format(a=a, b=b)
    run_simulation(output_filename=filename, a=a, b=b)
