#! /bin/bash
#SBATCH -A Berzelius-2023-222
#SBATCH --gpus 1
#SBATCH -N 1
#SBATCH -t 10:30:00
#SBATCH --mail-type ALL
#SBATCH --mail-user yuningw@kth.se
#SBATCH --output /proj/deepsep/users/x_yunwa/mst_pinn/mi_channel/finetuning/logs/fine-tune.out
#SBATCH --error /proj/deepsep/users/x_yunwa/mst_pinn/mi_channel/finetuning/logs/fine-tune.error

cd /proj/deepsep/users/x_yunwa/mst_pinn/mi_channel/finetuning
singularity run --nv tensorflow_2.6.1-gpu.sif python a2_weight_tune.py