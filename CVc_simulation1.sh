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
python src/frequentist_simulation/simulation.py  --num_fixed_snps=50 --save_path=./simulation_output/frequentist_CVc/using_fastlmm
python src/frequentist_simulation/simulation.py  --num_fixed_snps=10 --save_path=./simulation_output/frequentist_CVc/using_fastlmm

python src/frequentist_simulation/simulation.py  --num_fixed_snps=150 --save_path=./simulation_output/frequentist_CVc/using_fastlmm
python src/frequentist_simulation/simulation.py  --num_fixed_snps=100 --save_path=./simulation_output/frequentist_CVc/using_fastlmm
python src/frequentist_simulation/simulation.py  --num_fixed_snps=500 --save_path=./simulation_output/frequentist_CVc/using_fastlmm
python src/frequentist_simulation/simulation.py  --num_fixed_snps=200 --save_path=./simulation_output/frequentist_CVc/using_fastlmm

end=$SECONDS
echo "duration: $((end-start)) seconds."
