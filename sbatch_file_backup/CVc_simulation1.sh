#!/bin/bash
####### Reserve computing resources #############
#SBATCh --job-name=CVcSimulation
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --mem=16G
#SBATCH --partition=parallel
#SBATCH --error=out_simulation/%x_%j.out
#SBATCH --output=out_simulation/%x_%j.out

####### Set environment variables ###############
module load python/anaconda3-2019.10-tensorflowgpu
echo "load succeed"

####### Run your script #########################
echo "Start to Running"
start=$SECONDS
python src/frequentist_simulation/simulation.py  --num_fixed_snps=100 --num_large_effect=100 --save_path=./simulation_output/frequentist_CVc/using_fastlmm_100 --simulation_times=100

end=$SECONDS
echo "duration: $((end-start)) seconds."
