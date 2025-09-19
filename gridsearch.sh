#!/usr/bin/bash

virus="nCOV19_clinical_simulated_2"
primers="artic_v3_all_alt"

# Parameter arrays
c_values=(100 150 200)
v_values=(1 3 5)
b_values=(0.99 0.999 1)
decay_values=(2.5 3)

mkdir ../Master-Thesis-Project/data/gridsearch/${virus}
mkdir ../Master-Thesis-Project/data/MWU/${virus}

for c in "${c_values[@]}"; do
    for v in "${v_values[@]}"; do
        for b in "${b_values[@]}"; do
            for decay in "${decay_values[@]}"; do
                sh simulate_analyse.sh "$virus" "$primers" "$c" "$v" "$b" "$decay"
                # otherwise the next iter won't run
                rm -r ../Master-Thesis-Project/data/fastq_files/${virus}

                mv ../Master-Thesis-Project/data/primer_mismatch_dfs/${virus}.csv ../Master-Thesis-Project/data/gridsearch/${virus}_${c}_${v}_${decay}.csv
                mv ../Master-Thesis-Project/data/gridsearch/amplicon_counts_${virus}_dataset.csv ../Master-Thesis-Project/data/gridsearch/${virus}/amplicon_counts_${virus}_${c}_${v}_${decay}_dataset.csv
                mv ../Master-Thesis-Project/data/plots/${virus}/SNVs_abundance_bins_boxplot.png ../Master-Thesis-Project/data/plots/${virus}/${c}_${v}_${decay}.png
                mv ../Master-Thesis-Project/data/MWU/${virus}.md ../Master-Thesis-Project/data/MWU/${virus}/${c}_${v}_${decay}.md
            done
        done
    done
done
