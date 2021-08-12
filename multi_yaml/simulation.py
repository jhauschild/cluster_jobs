"""Example simulation file, containing a function that (usually) runs for a very long time...

To run your own simulations, adjust the functions by your liking,
but keep (or copy & paste) the last 3 lines of this file."""

from __future__ import print_function
import numpy as np
import pickle
import time


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
    for i in range(3):
        try:
            print("step ", i, flush=True)  # (remove `flush=True` for Python 2)
            # the flush=True makes the output appear immediately
        except TypeError:  # flush is not available for Python 2
            print("step ", i)
        time.sleep(1)

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
    import cluster_jobs
    cluster_jobs.run_simulation_commandline(globals())
