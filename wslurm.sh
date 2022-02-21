run_experiments \
  --cache_directory=/home/mila/c/chris.emezue/genedisco/cache2  \
  --output_directory=/home/mila/c/chris.emezue/genedisco/output2  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --max_num_jobs=1 \
  --schedule_on_slurm \
  --schedule_children_on_slurm \
  --remote_execution_virtualenv_path=/home/mila/c/chris.emezue/genedisco/genv
