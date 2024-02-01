"""Example simulation file, containing a function that (usually) runs for a very long time...

To run your own simulations, adjust the functions to your liking,
but keep (or copy & paste) the last 3 lines of this file."""

from __future__ import print_function
import numpy as np
import pickle
import time


def run_simulation(**kwargs):
    """Example dummy simulation.

    You could imagine this function to run a costly simulation (e.g. DMRG).

    Parameters to this function are arbitrary kewyword-arguments,
    which you can read out from the dictionary.
    """
    print("executing run_simulation() in file", __file__)
    print("got the dictionary kwargs =", kwargs)

    # HERE is where you would usually run your simulation (e.g. DMRG).
    # in this dummy simulation, we mimic some heavy calculations:
    # we draw roughly `use_GB` gigabytes of random numbers, and sum them up.
    run_mins = kwargs.get('run_mins', 0.5)
    use_GB = kwargs.get('use_GB', 1.)
    GB_size = int(1024**3 *8/64)  # assuming 64-bit numbers
    total_size = int(use_GB*GB_size)

    sums = []
    start_time = time.time()
    while time.time() - start_time < 60. * run_mins:
        random_numbers = np.random.random(size=total_size)
        y = np.sum(random_numbers)
        print("example: sum of random numbers =", y)
        time.sleep(10.)  # wait to give time to check memory usage while using all of it
        del random_numbers  # don't duplicate memory usage when initializing in next loop
        sums.append(y)

    results = {'kwargs': kwargs, 'sums': sums, 'average': np.mean(sums)}
    print("produced results =" , results)

    output_filename = kwargs['output_filename']
    print("save results to ", output_filename)
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)


def another_simulation(output_filename, a, b, sub_params):
    """Another example simulation."""
    print("executing another_simulation() in file", __file__)
    print("got arguments a={a!s},b={b!s},sub_params={sub!s}".format(a=a, b=b, sub=sub_params))
    # HERE is where you would usually run your simulation (e.g. some time evolution).
    time.sleep(30)  # simulate simulation: wait for 30 second
    results = {'example_data': np.random.random((3, 3))}

    print("save results to ", output_filename)
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # allow to call this script like `python simulation.py <RUN_MINS> <USE_GB>`
    import sys
    if len(sys.argv) == 3:
        run_mins = int(sys.argv[1])
        use_GB = float(sys.argv[2])
    else:
        run_mins = 10
        use_GB = 2.
    filename = "output_simple_run_{run_mins:d}m_use_{use_GB:.1f}GB.pkl"
    filename = filename.format(run_mins=run_mins, use_GB=use_GB)
    run_simulation(output_filename=filename, run_mins=run_mins, use_GB=use_GB)
