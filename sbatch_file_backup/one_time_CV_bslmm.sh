#!/bin/bash
####### Reserve computing resources #############
#SBATCh --job-name=oneSimulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
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
python src/bslmm_simulation/bslmm_simulation.py --num_fixed_snps=15 --simulation_times=1
end=$SECONDS
echo "duration: $((end-start)) seconds."
