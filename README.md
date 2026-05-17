# MASSiVe

This project is intended to simulate metagenomic amplicon sequenced reads for viruses. Simulated mixtures of amplicons are produced, based on proportions of viral genomes that are supplied by the user and a supported primer set of choice. See [my thesis](https://repository.tudelft.nl/record/uuid:2991e02f-d1ca-4738-be71-2c9ac09417fb) to learn about how it works.

## Installation

1. Clone this repository.

```sh
git clone https://github.com/ZoyavanMeel/MASSiVe.git
cd MASSiVe
```

2. Use conda as follows to create a new environment with the dependencies and correct versions (RECOMMENDED):

```sh
conda env create -f MASSiVe-env.yaml
conda activate MASSiVe
```

## Quickstart

This creates a synthetic metagenome from the fasta files in the example/genomes folder, using relative genome proportions from the example/abundances.tsv. The primers used are from the ARTIC protocol v3.

```sh
python src/simulate_metagenome.py --primer_set c --primers_file primer_sets/artic_v3_all_alt.fastq --primer_bed primer_sets/artic_v3_all_alt.bed --genomes_file test/SRR15711279.fa --genome_abundances test/abundances.tsv --output_folder simulation_output --temp_folder example/temp/ -r ref/MN908947.3.fasta
```

See `config.yaml` for recommended settings for clinical SARS-CoV-2 samples and Dengue virus.

### See the help page

```sh
python src/simulate_metagenome.py --help
```

## Output files

- example_R1.fastq & example_R2.fastq: simulated reads.
- example_amplicon_abundances_summary.tsv: a table summarising the amplicons.
- example_hf_errors.vcf: all the intended high-frequency errors. Observed VAFs may be different from those in the VCF file due to randomness and recurrence.
- example.log: The log file.

## Extra options and potential bugs

Things to watch out for:

- If you want to run multiple instances of MASSiVe simultaneously, make sure to use a unique `--TEMP_FOLDER` for each run. Otherwise, they will interfere with each other. Submitting multiple SLURM or LSF jobs or using SWAMPy in a Snakemake rule are examples of situations where this warning may apply.

- First and last amplicons get dropped for basically every genome except the Wuhan reference.
This is because the leftmost primer 1 and rightmost primer 98 basically never match in a genome.
Also watch out for long runs of N's in the primer sites, if these are there then that amplicon will drop out.

- If a source genome amplicon is has deletions at the end, PCR errors that were supposed to be on those loci will be skipped only for that souce genome. A warning is printed in this case.

- Special characters in the fasta genome ids could potentially cause a problem. In the code, the characters "/", "," and " " are dealt with,
but other whitespace characters or special characters such as "&" could cause a bug.

- Note that the reads in the fastq files are both shuffled by a randomly chosen permutation.
