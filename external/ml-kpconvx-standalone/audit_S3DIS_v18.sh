#!/bin/bash
# Audit graph changes before any V18 training.

cd KPConvX
export PYTHONPATH=$PWD:$PYTHONPATH

python3 experiments/S3DIS/audit_S3DIS_v18.py \
  --dataset_path "$PWD/../data/s3dis" \
  --profile v18_diagnostic \
  --samples 20 \
  --output "$PWD/results/v18_graph_audit.json"
