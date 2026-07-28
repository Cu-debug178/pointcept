#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

# Go to the root directory
cd KPConvX
export PYTHONPATH=$PWD:$PYTHONPATH

############
# Parameters
############

# Change the path to your dataset here
ARGS="--dataset_path $PWD/../data/ScanObjectNN/main_split"

# Choose the log path here
LOG_PATH="$PWD/results/ScanObjectNN_KPConvD-L"
ARGS="$ARGS --log_path $LOG_PATH"

# # Optionally, you can choose a specific weight file
# ARGS="$ARGS --weight_path $LOG_PATH/checkpoints/current_chkp.tar"

# If you provide the weight path, it has to be in the log_path folder. 
# It allows you to choose a specific weight file from the log folder.


###############
# Main function
###############

# Define the experiment and script to run
EXP="ScanObjectNN"
SCRIPT="test_ScanObj.py"

# Start the training
python3 experiments/$EXP/$SCRIPT $ARGS

