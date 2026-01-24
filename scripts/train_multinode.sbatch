#!/bin/bash
#SBATCH --job-name=i2sb-train
#SBATCH --partition=gpu-teaching-2d
#SBATCH --constraint='80gb'
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -e  # Exit on error

# --- ARGUMENT PARSING ---
# $1 is the first argument passed to sbatch
# Usage: sbatch script.sh path/to/config.yaml

if [[ -z "$1" ]]; then
    echo "Error: No config file provided."
    echo "Usage: sbatch $0 <path_to_config>"
    exit 1
fi

CONFIG_PATH="$1"

# Check if file exists before waiting for queue
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file '$CONFIG_PATH' not found!"
    exit 1
fi
# ------------------------

# NCCL optimizations for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0              # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback and docker interfaces
export NCCL_TIMEOUT=1800

# Prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# More actionable distributed error logs (especially when one rank dies).
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# Get the master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Use a job-specific port to reduce collisions when multiple jobs share a node.
MASTER_PORT=$((29500 + (${SLURM_JOB_ID:-0} % 1000)))

# Detect GPUs available *to this job*.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _CVD_ARR <<< "${CUDA_VISIBLE_DEVICES}"
    GPUS_PER_NODE=${#_CVD_ARR[@]}
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    IFS=',' read -ra _SJG_ARR <<< "${SLURM_JOB_GPUS}"
    GPUS_PER_NODE=${#_SJG_ARR[@]}
else
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi

# MIG / Duplicate PCI Bus ID handling
if [[ -n "${SLURM_JOB_GPUS:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    _BUS_IDS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | \
        awk -F', *' -v ids="${SLURM_JOB_GPUS}" '
            BEGIN { n=split(ids, a, ","); for (i=1; i<=n; i++) want[a[i]]=1 }
            ($1 in want) { print $2 }
        ')
    if [[ -n "${_BUS_IDS:-}" ]]; then
        _UNIQ_BUS_COUNT=$(printf "%s\n" "${_BUS_IDS}" | sort -u | wc -l)
        if [[ "${_UNIQ_BUS_COUNT}" -gt 0 && "${_UNIQ_BUS_COUNT}" -lt "${GPUS_PER_NODE}" ]]; then
            echo "[warn] SLURM allocated ${GPUS_PER_NODE} GPU(s) (${SLURM_JOB_GPUS}) but only ${_UNIQ_BUS_COUNT} unique PCI bus ID(s) were found."
            echo "[warn] Reducing torchrun --nproc_per_node to ${_UNIQ_BUS_COUNT} to avoid NCCL duplicate-GPU abort on MIG." 
            GPUS_PER_NODE=${_UNIQ_BUS_COUNT}
        fi
    fi
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_PATH"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "=========================================="

# Export variables for use inside container
export MASTER_ADDR MASTER_PORT GPUS_PER_NODE
export SLURM_NNODES SLURM_JOB_ID

# Launch with srun
# We use "$CONFIG_PATH" here. 
# Bash replaces the variable before passing the command to apptainer.
srun --export=ALL apptainer run \
    --bind /home/space/datasets/imagenet/2012:$PWD/datasets/imagenet \
    --nv \
    pml.sif \
    torchrun \
        --log-dir=./logs/torchrun_${SLURM_JOB_ID} \
        --tee=3 \
        --nnodes=${SLURM_NNODES:-1} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        main.py --config "$CONFIG_PATH"
