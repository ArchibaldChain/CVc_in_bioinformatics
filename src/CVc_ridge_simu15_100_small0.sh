#!/bin/bash
####### Reserve computing resources #############
#SBATCh --job-name=CVcRidge
#SBATCH --nodes=1
#SBATCH --ntasks=3
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
start_time=$(date  +'%Y-%m-%d %H:%M:%S')
echo "start time: $start_time"

python src/cross_validation_simulation.py --small_effect=0 --method=ols --alpha=10 --correcting --num_fixed_snps=100 --num_large_effect=15 --save_path=./simulation_output/frequentist_CVc/15_100_small0 --simulation_times=100
python src/cross_validation_simulation.py --small_effect=0 --method=gls --alpha=10 --correcting --num_fixed_snps=100 --num_large_effect=15 --save_path=./simulation_output/frequentist_CVc/15_100_small0 --simulation_times=100
python src/cross_validation_simulation.py --small_effect=0 --method=blup --alpha=10 --correcting --num_fixed_snps=100 --num_large_effect=15 --save_path=./simulation_output/frequentist_CVc/15_100_small0 --simulation_times=100 



end=$SECONDS

echo "duration: $((end-start)) seconds."
echo "duration: $(((end-start)/60)) mins."
