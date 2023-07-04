#!/bin/bash
####### Reserve computing resources #############
#SBATCh --job-name=CVcNonReg
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

python src/cross_validation_simulation.py --method=ols --correcting --num_fixed_snps=100 --num_large_effect=100 --save_path=./simulation_output/frequentist_CVc/usemaf --simulation_times=20
python src/cross_validation_simulation.py --method=gls --correcting --num_fixed_snps=100 --num_large_effect=100 --save_path=./simulation_output/frequentist_CVc/usemaf --simulation_times=20
python src/cross_validation_simulation.py --method=blup --correcting --num_fixed_snps=100 --num_large_effect=100 --save_path=./simulation_output/frequentist_CVc/usemaf --simulation_times=20

end=$SECONDS

echo "duration: $((end-start)) seconds."
echo "duration: $(((end-start)/60)) mins."
