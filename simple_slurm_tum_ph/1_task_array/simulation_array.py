"""Example simulation file, containing a function that (usually) runs for a very long time...

To run your own simulations, adjust the functions to your liking,
but keep (or copy & paste) the last 3 lines of this file."""

import simulation

# one dict of parameters for each array job
array_parameters = [
    dict(a=a, run_mins=run_mins, use_GB=1.5*a)
    for run_mins in [3, 4]
    for a in [1, 2]
]

if __name__ == "__main__":
    # allow to call this script like `python simulation.py <TASK_ID>`
    import sys
    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
        parameters = array_parameters[task_id]
    else:
        print("ERROR: need one argument TASK_ID")
        sys.exit(1)
    filename = "output_simple_a_{a:d}_run_{run_mins:.1f}_use_{use_GB:.1f}GB.pkl"
    parameters['output_filename'] = filename.format(**parameters)
    simulation.run_simulation(**parameters)
