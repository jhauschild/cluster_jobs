#!/bin/bash

echo "python {cluster_jobs_dir}/{cluster_jobs_py} run {config_file} {task_id}"
python {cluster_jobs_dir}/{cluster_jobs_py} run {config_file} {task_id}
