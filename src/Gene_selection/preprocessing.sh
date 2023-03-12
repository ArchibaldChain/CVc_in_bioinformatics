#!/bin/bash
####### Reserve computing resources #############
#SBATCh --job-name=oneSimulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=8G
#SBATCH --partition=parallel
#SBATCH --error=out/%x_%j.out
#SBATCH --output=out/%x_%j.out

####### Set environment variables ###############
module load python/anaconda3-2018.12
echo "load succeed"

####### Run your script #########################
echo "Start to Running"
start=$SECONDS
python vcf_preprocessing.py
end=$SECONDS
echo "duration: $((end-start)) seconds."
