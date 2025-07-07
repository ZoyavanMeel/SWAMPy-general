from numpy.random import dirichlet, binomial, multinomial
import numpy as np
import pandas as pd
import logging


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def exact_sampler(
    amplicon_df: pd.DataFrame,
    amplicon_distribution_file: str,
) -> pd.DataFrame:
    """
    Determines number of reads per amplicon as listed in `amplicon_distribution_file`:
    | amplicon_number | alt_num_left | alt_num_right | Genome_1 | Genome_2 | Genome_3 |
    | --------------: | -----------: | ------------: | -------: | -------: | -------: |
    | 1               | 1            | 1             | 100      | 300      | 400      |
    | 2               | 1            | 1             | 240      | 450      | 300      |
    | 3               | 1            | 1             | 240      | 360      | 100      |
    | 3               | 1            | 2             | 230      | 300      | 0        |
    | 3               | 2            | 1             | 240      | 0        | 100      |
    | 3               | 2            | 2             | 0        | 300      | 10       |
    """

    amplicon_df["hyperparameter"] = -1
    amplicon_df["genome_n_reads"] = -1
    amplicon_df["amplicon_prob"] = -1

    amp_dist_df = pd.read_csv(amplicon_distribution_file, sep="\t")

    # Z: make amplicon_distribution_file consistent
    if "alt_num_left" not in amp_dist_df.columns:
        amp_dist_df["alt_num_left"] = 1
    if "alt_num_right" not in amp_dist_df.columns:
        amp_dist_df["alt_num_right"] = 1

    amp_cols = ["amplicon_number", "alt_num_left", "alt_num_right"]
    amp_dist_df.columns = [
        # process the lineage names the same as before
        col.replace(" ", "_")  # .replace("/", "&").replace(",", "&")
        if col not in amp_cols
        else col for col in amp_dist_df.columns
    ]
    lineages = [col for col in amp_dist_df.columns if col not in amp_cols]

    amp_dist_df_long = amp_dist_df.melt(
        id_vars=amp_cols, var_name="ref",
        value_vars=lineages, value_name="n_reads"
    )

    amplicon_df = amplicon_df.merge(
        amp_dist_df_long,
        on=["ref", "amplicon_number", "alt_num_left", "alt_num_right"],
        how="left"
    )

    amplicon_df["total_n_reads"] = amp_dist_df[lineages].sum().sum()
    return amplicon_df


def dirichlet_sampler(
    dist: str,
    amplicon_df: pd.DataFrame,
    amplicon_distribution_file: str,
    amplicon_pseudocounts_c: int,
    genome_abundances: dict,
    total_n_reads: int
) -> pd.DataFrame:
    """
    I have implemented this the way I have for reproducibility. This method produces the exact same outputs as SWAMPy
    The implementation can be simplified and sped up by vectorizing more operations with Pandas, but I purposefully chose not to.
    Besides, the increased speed would likely be only marginal.
    """

    amplicon_df["total_n_reads"] = total_n_reads
    hyperparams = get_hyperparams(amplicon_distribution_file)
    genome_counts = get_amplicon_count_per_genome(total_n_reads, genome_abundances)
    amplicon_df["genome_n_reads"] = amplicon_df["ref"].map(genome_counts)

    # Z: effectively, the difference between Dirichlet model 1 and 2 is
    # whether you pull from the distribution...
    if dist == "DIRICHLET_1":
        hyperparams["amplicon_prob"] = dirichlet(hyperparams["hyperparameter"] * float(amplicon_pseudocounts_c))

    amplicon_df = amplicon_df.merge(hyperparams, on=["amplicon_number", "alt_num_left", "alt_num_right"], how="left")

    # this for-loop can be vectorized away
    amplicon_counts = {"ref": [], "amplicon_prob": []}  # , "n_reads": []}
    for ref in sorted(genome_abundances.keys()):
        amplicon_counts["ref"].extend([ref] * hyperparams.shape[0])

        if dist == "DIRICHLET_1":  # ... *once* for all genomes...
            amplicon_counts["n_reads"].extend(multinomial(genome_counts[ref], hyperparams["amplicon_prob"]))
        else:  # ... or *separately* for each genome
            amplicon_prob = dirichlet(hyperparams["hyperparameter"] * float(amplicon_pseudocounts_c))
            amplicon_counts["amplicon_prob"].extend(amplicon_prob)
            # amplicon_counts["n_reads"].extend(binomial(round(total_n_reads * genome_abundances[ref]), amplicon_prob))

    n_genomes = len(genome_abundances.keys())
    amplicon_counts["amplicon_number"] = hyperparams["amplicon_number"].tolist() * n_genomes
    amplicon_counts["alt_num_left"] = hyperparams["alt_num_left"].tolist() * n_genomes
    amplicon_counts["alt_num_right"] = hyperparams["alt_num_right"].tolist() * n_genomes

    amplicon_df = amplicon_df.merge(
        pd.DataFrame(amplicon_counts),
        on=["ref", "amplicon_number", "alt_num_left", "alt_num_right"]
    )

    # this apply is just a fancy-looking for-loop that can also go away, no need to call the binomial function separately for each row
    amplicon_df["n_reads"] = amplicon_df.apply(
        lambda x: binomial(round(x["total_n_reads"] * x["abundance"]), x["amplicon_prob"]),
        axis=1
    )
    return amplicon_df


def get_hyperparams(path: str) -> pd.DataFrame:
    """For each amplicon, look up what the dirichlet hyperparameter should be (parameter alpha)"""
    hyperparams = pd.read_csv(path, sep="\t")
    if "alt_num_left" not in hyperparams.columns:
        hyperparams["alt_num_left"] = 1
    if "alt_num_right" not in hyperparams.columns:
        hyperparams["alt_num_right"] = 1
    return hyperparams


def get_amplicon_count_per_genome(total_n_reads: int, genome_abundances: dict) -> dict:
    """For each genome, sample a total number of reads that should be shared between all of its amplicons:
    `N_genome = Multinomial(N_reads, p_genomes)`"""
    genome_counts = multinomial(total_n_reads, [genome_abundances[i] for i in sorted(genome_abundances.keys())])
    return {k: genome_counts[i] for i, k in enumerate(sorted(genome_abundances.keys()))}


def apply_amplicon_reads_sampler(
    amplicon_df: pd.DataFrame,
    amplicon_distribution: str,
    amplicon_distribution_file: str,
    amplicon_pseudocounts_c: int,
    genome_abundances: dict,
    total_n_reads: int
) -> pd.DataFrame:
    if amplicon_distribution.upper() == "EXACT":
        return exact_sampler(
            amplicon_df,
            amplicon_distribution_file
        )
    elif amplicon_distribution.upper() in ["DIRICHLET_1", "DIRICHLET_2"]:
        return dirichlet_sampler(
            amplicon_distribution,
            amplicon_df,
            amplicon_distribution_file,
            amplicon_pseudocounts_c,
            genome_abundances,
            total_n_reads
        )
    else:
        logging.info("Amplicon distribution not recognised, pick one of EXACT, DIRICHLET_1, DIRICHLET_2.")
        exit(1)
