#!/bin/bash
#SBATCH --job-name=i2sb-train
#SBATCH --partition=gpu-teaching-2d
#SBATCH --gpus=3
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -e  # Exit on error

# NCCL optimizations for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0              # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback and docker interfaces
export NCCL_TIMEOUT=1800

# Prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get the master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Detect GPUs available to this job
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node (this node): $GPUS_PER_NODE"
echo "Node list: $SLURM_JOB_NODELIST"
echo "=========================================="

# Export variables for use inside container
export MASTER_ADDR MASTER_PORT GPUS_PER_NODE
export SLURM_NNODES SLURM_JOB_ID

# Launch with srun
srun --export=ALL apptainer run \
    --bind /home/space/datasets/imagenet/2012:$PWD/datasets/imagenet \
    --nv \
    pml.sif \
    torchrun \
        --nnodes=${SLURM_NNODES:-1} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        main.py --config ./configs/superres-bicubic.yaml
