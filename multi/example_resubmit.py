"""Resubmit jobs for simulations of a `config` where the output file is missing.

If some job(s) got killed, e.g. because they would have needed a sligthly larger runtime,
you might want to resubmit basicaly the same configuration, selecting only the jobs which were not
yet successfully run.

I found it good practise to include the output filenames in the `kwargs` of any simulation,
and therefore into the job `config`. Thus, we can simply scan which files of a job config are
missing and resubmit the corresponding simulations, as is done below.

LIMITATION: if there are still some jobs running on the cluster to produce the output of this job
`config`, this script will restart new jobs producing the same output...
"""

import sys
import jobs_tum

if len(sys.argv) < 2:
    raise ValueError("Expect arguments: CONFIG_FILENAME [RESUBMIT_TASK_ID ...]")
config_filename = sys.argv[1]
config = jobs_tum.read_config_file(config_filename)
# here, you can update hardware requirements
#  config['require'].update(cpu='24:00:00')  # e.g. increase allowed runtime

# update jobname to indicate a re-submitted job
if not '_rerun' in config['jobname']:
    config['jobname'] = config_filename[:-len("_config.pkl")] + '_rerun'

if len(sys.argv) == 2:
    # find files where the output is missing
    missing = jobs_tum.output_missing(config)
    if len(missing) == 0:
        print("No output file missing")
        exit(1)
    resubmit_ids = [m+1 for m in missing]
    print("missing", len(resubmit_ids), "output files")
else:
    # got some numbers for job indices to resubmit
    resubmit_ids = [int(i) for i in sys.argv[2:]]

print("resubmit for the following previous SGE_TASK_IDs:")
print(resubmit_ids)

assert all([i > 0 for i in resubmit_ids])  # SGE_TASK_IDs start counting with 1, not 0.
# filter config['params']  to include only the simulations for the desired jobd ids
config['params'] = [config['params'][i-1] for i in resubmit_ids]

jobs_tum.submit_sge(config)
