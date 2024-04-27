#!/bin/bash

# Check the number of arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_path> <output_path> <model_path> [gpu_id] [ncpus]"
    exit 1
fi

# Assign the input parameters to variables
INPUT_PATH=$(realpath $1)
OUTPUT_PATH=$(realpath $2)
MODEL_PATH=$(realpath $3)
GPU_ID=${4:-0}  # Default to GPU 0 if not specified
ncpus=${5:-1}  # Default to 1 CPU if not specified

# Construct bind paths
BIND_PATHS="$INPUT_PATH:/input,$OUTPUT_PATH:/output,$MODEL_PATH:/model"

# Construct the GPU option
GPU_OPTION=""

# Check if a specific GPU ID is requested and handle it
if [ -n "$4" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    GPU_OPTION="--nv"
fi

apptainer run $GPU_OPTION --bind $BIND_PATHS unet.sif /input /output /model $ncpus
