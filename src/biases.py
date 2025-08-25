import numpy as np
import pandas as pd
import logging
import os

import helpers as hp


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


DECAY_CONST = 4


def linear_decay(x): return 1/(x+1)
def exp_decay(x, const=1): return np.exp(-x * const)
def linear_decay(x, slope=1, intercept=20): return intercept - slope*x


def apply_bias(
    amplicon_df: pd.DataFrame, temp_folder: str, rng: np.random.Generator,
    total_n_reads: int, genome_abundances: dict, amplicon_dirichlet_parameter: int,
    snv_dirichlet_parameter: int, snv_balance: float
) -> pd.DataFrame:
    """
    You have the SNVs that were introduced by the PCR proces
    You have the SNVs that were originally in the lineages (implement)

    for each variation of each amplicon in each genome:
        has error in primer? -> lower read count
        keep track of the subtraction in read count
        redistribute subtraction over amplicons for this genome
    check how many reads off-target the sum
        if over target: ignore
        if under target: add read count
    """
    amplicon_df = add_persistent_mutation_count(amplicon_df, os.path.join(temp_folder, "SNV"))
    amplicon_df.reset_index(inplace=True, drop=True)

    amplicon_df["total_n_reads"] = total_n_reads
    genome_counts = get_amplicon_count_per_genome(total_n_reads, genome_abundances, rng)
    amplicon_df["genome_n_reads"] = amplicon_df["ref"].map(genome_counts)

    unique_snvs = sorted(amplicon_df["SNVs_in_primers"].unique())
    unique_refs = amplicon_df["ref"].unique()

    # Construct SNV dictionary
    snv_dict: dict[str, dict[int, float]] = {
        "hyperparams": {group: exp_decay(group, DECAY_CONST) for group in unique_snvs}
    }
    props = rng.dirichlet(
        [hp * float(snv_dirichlet_parameter) for hp in snv_dict["hyperparams"].values()]
    )
    snv_dict["prop"] = {group: prop for group, prop in zip(unique_snvs, props)}

    # Pull amplicon probabilities per genome from Dirichlet
    SNV_groups_to_indices = amplicon_df.groupby(["ref", "SNVs_in_primers"]).indices

    # Like SWAMPy's Dirichlet2 model
    amplicon_df["amplicon_prop"] = 0.0
    amplicon_df["n_reads"] = 0

    for genome in unique_refs:
        g_mask = amplicon_df["ref"] == genome
        amplicon_df.loc[g_mask, "amplicon_prop"] = rng.dirichlet(
            amplicon_df.loc[g_mask, "hyperparameter"] * float(amplicon_dirichlet_parameter)
        )

    for key, indices in SNV_groups_to_indices.items():
        genome, SNV_group = key
        # What fraction of the genome reads should this SNV group get according to the decay function?
        # -> snv_dict["prop"][group]

        # What fraction of the genome reads should this SNV group get according to the fraction of amplicons in this group?
        amp_read_props = amplicon_df.loc[indices, "amplicon_prop"]
        amp_read_prop = amp_read_props.sum()

        # The number of reads that is allotted to this SNV group is a balance between these two fractions
        snv_group_total = (
            # (b * SNV_prop) + ((1-b) * amp_prop) * n_genome
            (snv_balance) * snv_dict["prop"][SNV_group] + (1-snv_balance) * amp_read_prop
        ) * genome_counts[genome]

        # Normalise the amplicon proportion for read allocation per amp in this SNV group
        norm_amp_props = amp_read_props / amp_read_prop  # pd.Series / float

        allocations = rng.multinomial(snv_group_total, norm_amp_props)

        # Assign read allocations to corresponding indices
        amplicon_df.loc[indices, "n_reads"] = np.floor(allocations)
    return amplicon_df


def add_persistent_mutation_count(amplicon_df: pd.DataFrame, snv_folder: str) -> pd.DataFrame:
    SNV_files = os.listdir(snv_folder)
    for file in SNV_files:
        snv_df = pd.read_csv(os.path.join(snv_folder, file), index_col=False, delimiter="\t")
        if snv_df.empty:
            continue

        genome_mask = amplicon_df["ref"] == snv_df["REGION"].unique()[0]
        amps_in_genome = amplicon_df.loc[genome_mask]
        for _, snv in snv_df.iterrows():
            pos = snv["POS"]
            amplicon_df.loc[genome_mask, "SNVs_in_primers"] += (((
                pos >= amps_in_genome["left"]
            ) & (
                pos <= amps_in_genome["left"]+amps_in_genome["left_primer_length"]
            )) | ((
                pos >= amps_in_genome["right"]
            ) & (
                pos <= amps_in_genome["right"]+amps_in_genome["right_primer_length"]
            ))).astype(int)
    return amplicon_df


def adjust_to_requested(read_col: pd.Series, total_n_reads: int) -> pd.Series:
    allocated_reads = read_col.sum()
    scale = total_n_reads / allocated_reads
    return np.floor(read_col * scale)


def exact_sampler(
    amplicon_df: pd.DataFrame,
    amp_dist_df: str,
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
    amp_dist_df: pd.DataFrame,
    amplicon_dirichlet_parameter: int,
    genome_abundances: dict,
    total_n_reads: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    I have implemented this the way I have for reproducibility. This method produces the exact same outputs as SWAMPy
    The implementation can be simplified and sped up by vectorizing more operations with Pandas, but I purposefully chose not to.
    Besides, the increased speed would likely be only marginal.
    """

    amplicon_df["total_n_reads"] = total_n_reads
    genome_counts = get_amplicon_count_per_genome(total_n_reads, genome_abundances, rng)
    amplicon_df["genome_n_reads"] = amplicon_df["ref"].map(genome_counts)

    # Z: effectively, the difference between Dirichlet model 1 and 2 is
    # whether you pull from the distribution...
    if dist == "DIRICHLET_1":
        amp_dist_df["amplicon_prob"] = rng.dirichlet(
            amp_dist_df["hyperparameter"] * float(amplicon_dirichlet_parameter)
        )

    which = "n_reads" if dist == "DIRICHLET_1" else "amplicon_prob"
    amplicon_counts = {"ref": [], which: []}

    # this for-loop can be vectorized away
    for ref in sorted(genome_abundances.keys()):
        amplicon_counts["ref"].extend([ref] * amp_dist_df.shape[0])
        if dist == "DIRICHLET_1":  # ... *once* for all genomes...
            amplicon_counts["n_reads"].extend(rng.multinomial(
                genome_counts[ref], amp_dist_df["amplicon_prob"]))
        else:  # ... or *separately* for each genome
            amplicon_prob = rng.dirichlet(
                amp_dist_df["hyperparameter"] * float(amplicon_dirichlet_parameter))
            amplicon_counts["amplicon_prob"].extend(amplicon_prob)
            # amplicon_counts["n_reads"].extend(binomial(round(total_n_reads * genome_abundances[ref]), amplicon_prob))

    n_genomes = len(genome_abundances.keys())
    amplicon_counts["amplicon_number"] = amp_dist_df["amplicon_number"].tolist() * n_genomes
    amplicon_counts["alt_num_left"] = amp_dist_df["alt_num_left"].tolist() * n_genomes
    amplicon_counts["alt_num_right"] = amp_dist_df["alt_num_right"].tolist() * n_genomes

    amplicon_df = amplicon_df.merge(
        pd.DataFrame(amplicon_counts),
        on=["ref", "amplicon_number", "alt_num_left", "alt_num_right"]
    )
    amplicon_df = amplicon_df.merge(
        amp_dist_df.drop(columns=["hyperparameter"]),
        on=["amplicon_number", "alt_num_left", "alt_num_right"],
        how="left"
    )

    if dist == "DIRICHLET_2":
        amplicon_df["n_reads"] = amplicon_df.apply(
            lambda x: rng.binomial(round(x["total_n_reads"] * x["abundance"]), x["amplicon_prob"]),
            axis=1
        )
    return amplicon_df


def set_alt_nums(df: pd.DataFrame) -> pd.DataFrame:
    if "alt_num_left" not in df.columns:
        df["alt_num_left"] = 1
    if "alt_num_right" not in df.columns:
        df["alt_num_right"] = 1
    return df


def get_amplicon_count_per_genome(total_n_reads: int, genome_abundances: dict, rng: np.random.Generator) -> dict:
    """For each genome, sample a total number of reads that should be shared between all of its amplicons:
    `N_genome = Multinomial(N_reads, p_genomes)`"""
    genome_counts = rng.multinomial(
        total_n_reads, [genome_abundances[i] for i in sorted(genome_abundances.keys())])
    return {k: genome_counts[i] for i, k in enumerate(sorted(genome_abundances.keys()))}


def apply_amplicon_reads_sampler(
    amplicon_df: pd.DataFrame,
    amplicon_distribution: str,
    amp_dist_df: pd.DataFrame,
    amplicon_dirichlet_parameter: int,
    genome_abundances: dict,
    total_n_reads: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    if amplicon_distribution.upper() == "EXACT":
        return exact_sampler(
            amplicon_df,
            amp_dist_df
        )
    elif amplicon_distribution.upper() in ["DIRICHLET_1", "DIRICHLET_2"]:
        return dirichlet_sampler(
            amplicon_distribution,
            amplicon_df,
            amp_dist_df,
            amplicon_dirichlet_parameter,
            genome_abundances,
            total_n_reads,
            rng
        )
    else:
        logging.info("Amplicon distribution not recognised, pick one of EXACT, DIRICHLET_1, DIRICHLET_2.")
        exit(1)


def load_amp_dist_file(amplicon_distribution_file: str, primer_bed_path: str):
    if amplicon_distribution_file is not None:
        amp_dist_df = set_alt_nums(pd.read_csv(amplicon_distribution_file, sep="\t"))
    else:
        amp_dist_df = hp.read_primer_bed(primer_bed_path)[["amplicon_number", "alt_num_left", "alt_num_right"]]
        amp_dist_df["hyperparameter"] = 1/amp_dist_df.shape[0]
    return amp_dist_df
