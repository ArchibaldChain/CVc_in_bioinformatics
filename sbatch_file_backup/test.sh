#!/bin/bash
####### Reserve computing resources #############
#SBATCh --job-name=CVcSimulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu2022
#SBATCH --error=out_simulation/%x_%j.out
#SBATCH --output=out_simulation/%x_%j.out

####### Set environment variables ###############
module load python/anaconda3-2019.10-tensorflowgpu
echo "load succeed"

####### Run your script #########################
echo "Start to Running"
start=$SECONDS
python src/Test/test_gemma_Var.py
end=$SECONDS
echo "duration: $((end-start)) seconds."
