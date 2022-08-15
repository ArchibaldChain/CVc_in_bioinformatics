# Cross-Validation correction in Genomics Datasets

This project contains 3 parts

## Gene-Selection
- `vcf_preprocessing.py` is used to filter the SNP with MAF > 0.05.
- `gene_selection.py` is used to select genes from the preprocessed SNPs file
This project used SNPs in Chromosome 1 of `1000 Genomics Project` Dataset. The Dataset is not included in the directory because it is too large. Only the example dataset is included.
- And the selected genes are in the directory `data/SNP_in_200GENE_chr1.csv`.

-----
## FaST-LMM 
FaST-LMM is a implementation of [FaST linear mixed models for genome-wide association studies](https://www.nature.com/articles/nmeth.1681).

The documentation for FaST-LMM is available [here](FAST_LMM/Readme.md).

------
## Phenotype Simulation and Model Cross-Validation correction
We chose 2000 SNPs and simulated the large and weak effects of the SNPs. Then we got the phenotype. The datasets is saved as `data/Simulated_SNPs.csv`

We tested cross-validation correction using Linear Mixed Model, Weighted Least Square, Ordinary Least Square, and Ridge.

The calculation detail is available [here](Cross-validation correction Simulation Procedure.md)
