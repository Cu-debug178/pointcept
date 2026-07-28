#!/bin/bash
# Strong V18 mechanism diagnostic. This is not the final paper configuration.

cd KPConvX
export PYTHONPATH=$PWD:$PYTHONPATH

ARGS="--dataset_path $PWD/../data/s3dis"
ARGS="$ARGS --enable_dual_support 1"
ARGS="$ARGS --dual_support_profile v18_diagnostic"
ARGS="$ARGS --dual_support_debug 1"

python3 experiments/S3DIS/train_S3DIS.py $ARGS
