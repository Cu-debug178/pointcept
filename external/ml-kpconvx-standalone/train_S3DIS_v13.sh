#!/bin/bash
# V13 DA-Radius on the official KPConvX Standalone S3DIS recipe.

cd KPConvX
export PYTHONPATH=$PWD:$PYTHONPATH

ARGS="--dataset_path $PWD/../data/s3dis"
ARGS="$ARGS --enable_da_radius 1"
ARGS="$ARGS --da_radius_profile pointcept_v13"
ARGS="$ARGS --da_radius_debug 1"

python3 experiments/S3DIS/train_S3DIS.py $ARGS
