#!/usr/bin/bash
# Usage: ./simulate_many_samples.sh <num_runs> <master_seed>

NUM_RUNS=${1:-5}    # default 5 runs if not specified
MASTER_SEED=${2:-0} # default master seed if not specified

TOTAL_MIN=120000
TOTAL_MAX=220000

ABUNDANCE_FILE="../example/abundances" # .tsv

exec_simulater() {
    local i=$1
    SEED=$(python -c "import random; random.seed($MASTER_SEED+$i); print(random.randint(0, 99999))")
    TOTAL=$(python -c "import random; random.seed($MASTER_SEED+$i+9999); print(random.randint($TOTAL_MIN, $TOTAL_MAX))")

    echo "Run $i: --seed $SEED --total $TOTAL"

    python make_random_abundance.py -a "${ABUNDANCE_FILE}.tsv" -s "$SEED" -o "${ABUNDANCE_FILE}_${i}.tsv"
    python simulate_metagenome.py \
        --primer_set c \
        --primers_file ../primer_sets/artic_v3_all_alt.fastq \
        --primer_bed ../primer_sets/artic_v3_all_alt.bed \
        --genomes_file ../example/genomes.fasta \
        --genome_abundances "${ABUNDANCE_FILE}_${i}.tsv" \
        --output_folder "../simulation_output/run_$i" \
        --temp_folder "../example/temp/temp$1/" \
        --snv_balance 0.8 \
        -c 400 -v 200 \
        --seed "$SEED" \
        --autoremove \
        -n "$TOTAL" \
        --quiet
    mv "../simulation_output/run_$i/example_R1.fastq" "../../Master-Thesis-Project/data/fastq_files/nCOV19_clinical_simulated/example${i}_1.fastq"
    mv "../simulation_output/run_$i/example_R2.fastq" "../../Master-Thesis-Project/data/fastq_files/nCOV19_clinical_simulated/example${i}_2.fastq"
    rm "${ABUNDANCE_FILE}_${i}.tsv"
}

mkdir ../example/temp
for i in $(seq 1 $NUM_RUNS); do
    exec_simulater "$i" &
done
wait
rmdir ../example/temp
