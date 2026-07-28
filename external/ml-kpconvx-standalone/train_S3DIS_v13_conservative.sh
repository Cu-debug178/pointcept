#!/bin/bash
# Lower-risk V13 profile for the larger physical support of Standalone S3DIS.

cd KPConvX
export PYTHONPATH=$PWD:$PYTHONPATH

ARGS="--dataset_path $PWD/../data/s3dis"
ARGS="$ARGS --enable_da_radius 1"
ARGS="$ARGS --da_radius_profile standalone_conservative"
ARGS="$ARGS --da_radius_debug 1"

python3 experiments/S3DIS/train_S3DIS.py $ARGS
