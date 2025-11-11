#!/bin/bash
# Helper script to run robot control
# Usage: ./run_robot_control.sh [spot|stretch] [num_steps]

ROBOT=${1:-spot}
STEPS=${2:-50}

echo "=========================================="
echo "ROBOT CONTROL SCRIPT"
echo "Robot: $ROBOT"
echo "Steps: $STEPS"
echo "=========================================="

# Setup conda
source /scratch/tshu2/yyin34/projects/3d_belief/miniconda3/etc/profile.d/conda.sh
conda activate dfm-pixel-habitat

# Set library path
export LD_LIBRARY_PATH=/scratch/tshu2/yyin34/projects/3d_belief/miniconda3/envs/dfm-pixel-habitat/lib:$LD_LIBRARY_PATH

# Navigate to project root
cd /home/kli95/scratchtshu2/kli95/partnr-planner

# Run robot control
python -m habitat_llm.examples.robot_control --robot $ROBOT --steps $STEPS

