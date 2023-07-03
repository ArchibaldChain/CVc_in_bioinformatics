#!/bin/bash
####### Reserve computing resources #############
#SBATCh --job-name=BSLMMSimulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --mem=8G
#SBATCH --partition=parallel
#SBATCH --error=out_simulation/%x_%j.out
#SBATCH --output=out_simulation/%x_%j.out

####### Set environment variables ###############
module load python/anaconda3-2019.10-tensorflowgpu
echo "load succeed"

####### Run your script #########################
echo "Start to Running"
start=$SECONDS
start_time=$(date  +'%Y-%m-%d %H:%M:%S')
echo "start time: $start_time"
python src/bslmm_simulation/bslmm_simulation.py --simulation_times=1 --num_fixed_snps=100 --n_folds=5 --num_large_effect=100 --bslmm_save_path=./simulation_output/bslmm_CVc_sigma_100_fixed
end=$SECONDS
echo "duration: $(((end-start)/60)) mins."
