#!/bin/bash
#SBATCH --job-name=genedisco
#SBATCH --gres=gpu:20GB:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/genedisco/slurmerror_5.txt
#SBATCH --output=/home/mila/c/chris.emezue/genedisco/slurmoutput_5.txt

###########cluster information above this line
module load python/3.8
module load cuda/11.1/cudnn/8.0

source /home/mila/c/chris.emezue/genedisco/genv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mila/c/chris.emezue/genedisco/genv/lib"


CACHE_NAME=cache_umap_kde_run
OUTPUT_NAME=output_umap_kde_run
CACHE_DIR=/home/mila/c/chris.emezue/genedisco/${CACHE_NAME}
OUTPUT_DIR=/home/mila/c/chris.emezue/genedisco/${OUTPUT_NAME}

mkdir -p $CACHE_DIR
mkdir -p $OUTPUT_DIR

run_experiments \
  --cache_directory=${CACHE_DIR}  \
  --output_directory=${OUTPUT_DIR}  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --max_num_jobs=1 \
  --acquisition_function_path ../active_learning_methods/acquisition_functions/

#python -m pdb -c continue run_experiments_application.py --cache_directory=/home/mila/c/chris.emezue/genedisco/cache2 --output_directory=/home/mila/c/chris.emezue/genedisco/output2 --acquisition_batch_size=64 --num_active_learning_cycles=1 --max_num_jobs=1