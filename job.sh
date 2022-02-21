#!/bin/bash
#SBATCH --job-name=genedisco
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/genedisco/slurmerror_3.txt
#SBATCH --output=/home/mila/c/chris.emezue/genedisco/slurmoutput_3.txt

###########cluster information above this line
module load python/3.8
module load cuda/11.1/cudnn/8.0

source /home/mila/c/chris.emezue/genedisco/genv/bin/activate
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mila/c/chris.emezue/genedisco/genv/lib"

run_experiments \
  --cache_directory=/home/mila/c/chris.emezue/genedisco/cache  \
  --output_directory=/home/mila/c/chris.emezue/genedisco/output  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --max_num_jobs=1