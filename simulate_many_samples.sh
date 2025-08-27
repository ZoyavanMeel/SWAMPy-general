#!/usr/bin/bash
# Usage: ./simulate_many_samples.sh <num_runs> <master_seed> <clinical/wastewater>

NUM_RUNS=${1:-5}          # default 5 runs if not specified
MASTER_SEED=${2:-0}       # default master seed if not specified
CLINICAL=${3:-"clinical"} # default single-/meta-genomic

TOTAL_MIN=120000
TOTAL_MAX=220000

ABUNDANCE_FILE="../example/abundances" # .tsv

exec_simulater() {
    local i=$1
    LOG="../logs/log_$i.log"

    SEED=$(python -c "import random; random.seed($MASTER_SEED+$i); print(random.randint(0, 99999))")
    TOTAL=$(python -c "import random; random.seed($MASTER_SEED+$i+9999); print(random.randint($TOTAL_MIN, $TOTAL_MAX))")

    echo "Run $i: --seed $SEED --total $TOTAL" >>$LOG

    python make_random_abundance.py -a "${ABUNDANCE_FILE}.tsv" -s "$SEED" -c $CLINICAL -o "${ABUNDANCE_FILE}_${i}.tsv" >>$LOG
    python simulate_metagenome.py \
        --primer_set c \
        --primers_file ../primer_sets/artic_v3_all_alt.fastq \
        --primer_bed ../primer_sets/artic_v3_all_alt.bed \
        --genomes_file ../example/genomes.fasta \
        --genome_abundances "${ABUNDANCE_FILE}_${i}.tsv" \
        --output_folder "../simulation_output/run_$i" \
        --temp_folder "../example/temp/temp$1/" \
        --snv_balance 1 \
        -c 200 -v 500 \
        --seed "$SEED" \
        --autoremove \
        -n "$TOTAL" \
        --subs_VAF_alpha "9.28,0.945" \
        --del_VAF_alpha "4.72,0.41" \
        --ins_VAF_alpha "2.64,0.45" \
        --quiet >>$LOG

    mv "../simulation_output/run_$i/example_R1.fastq" "../../Master-Thesis-Project/data/fastq_files/nCOV19_clinical_simulated/example${i}_1.fastq"
    mv "../simulation_output/run_$i/example_R2.fastq" "../../Master-Thesis-Project/data/fastq_files/nCOV19_clinical_simulated/example${i}_2.fastq"
    rm "${ABUNDANCE_FILE}_${i}.tsv"
}

mkdir ../example/temp
for i in $(seq 1 $NUM_RUNS); do
    exec_simulater "$i" &
done
wait
rmdir ../example/temp --ignore-fail-on-non-empty
