#!/usr/bin/bash
VIRUS=$1
PRIMERS=$2
C=$3
V=$4
B=$5
DECAY=$6

# artic_v3_all_alt
echo "Simulating ${VIRUS}"
mkdir ../Master-Thesis-Project/data/fastq_files/$VIRUS
snakemake --profile slurm --snakefile snakefile_simulate all --config virus=${VIRUS} cli_c=${C} cli_v=${V} cli_b=${B} cli_decay=${DECAY}

# can contain empty files due to error handling
find ../Master-Thesis-Project/data/fastq_files/$VIRUS -type f -size 0 -print -delete

cd ../Master-Thesis-Project

echo "Processing FASTQ files"
snakemake --profile slurm --snakefile snakefile_processing calc_all_relative_abundance --config virus=$VIRUS

echo "Calling SNVs in primer regions"
snakemake --profile slurm --snakefile snakefile_snv all --config virus=$VIRUS

echo "Analysing SNV and amplification data"
mkdir data/plots/${VIRUS}
python src/statistical_testing.py -v $VIRUS -p data/primer_beds/${PRIMERS}.bed -s "" -t 1
