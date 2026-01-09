#!/bin/bash
#SBATCH --partition=gpu-teaching-7d
#SBATCH --gres=gpu:80gb:4
#SBATCH --job-name=i2sb_train
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# MPS should already be running on the cluster, so no need to start it

# Run the training command
srun --export=ALL apptainer run --bind /home/space/datasets/imagenet/2012:$PWD/datasets/imagenet --nv ../PML-I2SB/pml.sif accelerate launch --multi_gpu --num_processes 4 main.py --config configs/inpaint-freeform2030.yaml

# No cleanup needed since we didn't start MPS