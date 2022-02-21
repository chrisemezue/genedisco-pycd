#!/bin/bash
#SBATCH --job-name=genedisco_special_acquisition
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/genedisco/slurmerror_special_acquisition.txt
#SBATCH --output=/home/mila/c/chris.emezue/genedisco/slurmoutput_special_acquisition.txt

###########cluster information above this line
module load python/3.8
module load cuda/11.1/cudnn/8.0

source /home/mila/c/chris.emezue/genedisco/genv/bin/activate
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mila/c/chris.emezue/genedisco/genv/lib"


active_learning_loop  \
    --cache_directory=/home/mila/c/chris.emezue/genedisco/cache_acq  \
    --output_directory=/home/mila/c/chris.emezue/genedisco/output_acq  \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="custom" \
    --acquisition_function_path=/home/mila/c/chris.emezue/genedisco/special.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 

