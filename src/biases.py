from functools import partial
import numpy as np
import pandas as pd
import logging
import os

import helpers as hp


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def exp_decay(x, const=1): return np.exp(-x * const)
def linear_decay(x, slope=1, intercept=20): return intercept - slope*x
def sigmoid_decay(x): return 0.5 + 1 / (1 + np.exp(x))


def distribute_reads_among_amp_vars(group: pd.DataFrame, rng: np.random.Generator):
    if len(group["props"]) == 1:
        return group

    # total reads for this amplicon group (= {"n_reads": [n, 0, 0, 0, ...], "is_max": [True, False, False, False, ...]})
    n_reads = group["n_reads"].sum()
    props = group["props"].values
    # props = group["amplicon_prop"].values

    if props.sum() == 0:
        return group

    norm_props = props / props.sum()  # normalize for this group of variations

    group["n_reads"] = rng.multinomial(n_reads, norm_props)
    return group


def apply_bias(
    amplicon_df: pd.DataFrame, temp_folder: str, rng: np.random.Generator,
    total_n_reads: int, genome_abundances: dict, amplicon_dirichlet_parameter: float,
    snv_dirichlet_parameter: int, snv_balance: float, decay_const: float
) -> pd.DataFrame:
    amplicon_df = add_persistent_mutation_count(amplicon_df, os.path.join(temp_folder, "SNV1"))
    amplicon_df = add_persistent_mutation_count(amplicon_df, os.path.join(temp_folder, "SNV2"))
    amplicon_df.reset_index(inplace=True, drop=True)

    amplicon_df["total_n_reads"] = total_n_reads
    genome_counts = get_amplicon_count_per_genome(total_n_reads, genome_abundances, rng)
    amplicon_df["genome_n_reads"] = amplicon_df["ref"].map(genome_counts)

    amplicon_df["snv_alphas"] = snv_dirichlet_parameter * np.exp(-amplicon_df["SNVs_in_primers"] * decay_const)

    # four original+alt primer combos
    alts = amplicon_df.loc[amplicon_df["alt_num_left"] == 2, "amplicon_number"].drop_duplicates()
    amplicon_df.loc[amplicon_df["amplicon_number"].isin(alts), "snv_alphas"] /= 4

    amplicon_df["amp_alphas"] = amplicon_df["hyperparameter"] * amplicon_dirichlet_parameter

    amplicon_df["n_reads"] = 0

    unique_refs = amplicon_df["ref"].unique()

    for genome in unique_refs:
        g_mask = amplicon_df["ref"] == genome
        genome_amp_df: pd.DataFrame = amplicon_df.loc[g_mask].copy()

        # can't have alphas <= 0, so replacing them with 1e-24 is effectively the same
        amp_props = rng.dirichlet(genome_amp_df["amp_alphas"].replace(0, 1e-24))
        snv_props = rng.dirichlet(genome_amp_df["snv_alphas"].replace(0, 1e-24))
        props = snv_balance*snv_props + (1-snv_balance)*amp_props

        amplicon_df.loc[g_mask, "amp_props"] = amp_props
        amplicon_df.loc[g_mask, "snv_props"] = snv_props
        amplicon_df.loc[g_mask, "props"] = props

        # mark which amplicon variation has the highest proportionality
        # only those ones will get reads allocated to them.
        # afterwards, we divide the reads per amplicons over each variation
        groupby = amplicon_df.loc[g_mask].groupby(["amplicon_number", "alt_num_left", "alt_num_right"])
        is_max = amplicon_df.loc[g_mask, "props"] == groupby["props"].transform('max')

        max_props = amplicon_df.loc[(g_mask & is_max), "props"]
        norm_props = max_props / max_props.sum()

        read_allocations = rng.multinomial(genome_counts[genome], norm_props)
        amplicon_df.loc[(g_mask & is_max), "n_reads"] = read_allocations

    return amplicon_df


# def apply_bias(
#     amplicon_df: pd.DataFrame, temp_folder: str, rng: np.random.Generator,
#     total_n_reads: int, genome_abundances: dict, amplicon_dirichlet_parameter: int,
#     snv_dirichlet_parameter: int, snv_balance: float, decay_const: float
# ) -> pd.DataFrame:
#     amplicon_df = add_persistent_mutation_count(amplicon_df, os.path.join(temp_folder, "SNV1"))
#     amplicon_df = add_persistent_mutation_count(amplicon_df, os.path.join(temp_folder, "SNV2"))
#     amplicon_df.reset_index(inplace=True, drop=True)

#     amplicon_df["total_n_reads"] = total_n_reads
#     genome_counts = get_amplicon_count_per_genome(total_n_reads, genome_abundances, rng)
#     amplicon_df["genome_n_reads"] = amplicon_df["ref"].map(genome_counts)

#     unique_snvs = sorted(amplicon_df["SNVs_in_primers"].unique())
#     unique_refs = amplicon_df["ref"].unique()

#     # Construct SNV dictionary
#     # = {0: 1, 1: 0.0001, 2: 0.0}
#     snv_dict: dict[str, dict[int, float]] = {
#         "hyperparams": {group: exp_decay(group, decay_const) for group in unique_snvs}
#     }
#     props = rng.dirichlet(
#         [hp * float(snv_dirichlet_parameter) for hp in snv_dict["hyperparams"].values()]
#     )
#     snv_dict["prop"] = {group: prop for group, prop in zip(unique_snvs, props)}

#     # Pull amplicon probabilities per genome from Dirichlet
#     SNV_groups_to_indices = amplicon_df.groupby(["ref", "SNVs_in_primers"]).indices

#     amplicon_df["amplicon_prop"] = 0.0
#     amplicon_df["n_reads"] = 0

#     amplicon_df["is_max"] = False

#     for genome in unique_refs:
#         g_mask = amplicon_df["ref"] == genome

#         # assign proportion of genome's read count to each amplicon (variation)
#         amplicon_df.loc[g_mask, "amplicon_prop"] = rng.dirichlet(
#             amplicon_df.loc[g_mask, "hyperparameter"] * float(amplicon_dirichlet_parameter)
#         )

#         # mark which amplicon variation has the highest proportionality
#         # only those ones will get reads allocated to them.
#         # afterwards, we divide the reads per amplicons over each variation
#         amplicon_df.loc[g_mask, "is_max"] = amplicon_df.loc[g_mask, "amplicon_prop"].eq(
#             amplicon_df[g_mask].groupby(["amplicon_number", "alt_num_left", "alt_num_right"])["amplicon_prop"].transform('max'))

#     for key, indices in SNV_groups_to_indices.items():
#         genome, SNV_group = key
#         # What fraction of the genome reads should this SNV group get according to the decay function?
#         # -> snv_dict["prop"][group]

#         # What fraction of the genome reads should this SNV group get according to the fraction of amplicons in this group?
#         # amp_read_props = amplicon_df.loc[indices, "amplicon_prop"]
#         # only assign reads to the amplicon variations that occur in highest proportion

#         amp_read_props_all_vars = amplicon_df.loc[indices, ["amplicon_prop", "is_max"]]
#         amp_read_props_only_max = amp_read_props_all_vars.loc[amp_read_props_all_vars["is_max"]]["amplicon_prop"]

#         if amp_read_props_only_max.empty:
#             continue

#         # The number of reads that is allotted to this SNV group is a balance between these two fractions
#         snv_group_total = (
#             # (b * SNV_prop) + ((1-b) * amp_prop) * n_genome
#             (snv_balance) * snv_dict["prop"][SNV_group] + (1-snv_balance) *
#             amp_read_props_all_vars["amplicon_prop"].sum()
#         ) * genome_counts[genome]

#         # Normalise the amplicon proportion for read allocation per amp in this SNV group
#         norm_amp_props = amp_read_props_only_max / amp_read_props_only_max.sum()  # pd.Series / float

#         allocations = rng.multinomial(snv_group_total, norm_amp_props)

#         # Assign read allocations to corresponding indices
#         amplicon_df.loc[amp_read_props_only_max.index, "n_reads"] = np.floor(allocations)

#     return amplicon_df


def correct_dropout_rate(amplicon_df: pd.DataFrame, rate_mean: float, rate_std: float, rng: np.random.Generator) -> pd.DataFrame:
    unique_refs = amplicon_df["ref"].unique()
    indeces_to_drop = []
    for genome in unique_refs:
        g_mask = amplicon_df["ref"] == genome

        # all amplicons that got the max proportionality of their variation in this genome
        max_df = amplicon_df[amplicon_df["is_max"] & g_mask]
        dropped_mask = max_df["n_reads"] == 0
        n_curr_dropped_amps = dropped_mask.sum()
        curr_drop_rate = n_curr_dropped_amps / max_df.shape[0]

        if curr_drop_rate > rate_mean + 3*rate_std:
            logging.warning(
                f"{genome}: The current dropout rate is: {curr_drop_rate:.2f}. This is already >3 standard deviations (3*{rate_std}) larger than the given rate_mean ({rate_mean}). No extra amplicons will be dropped.")
            continue

        rate = rng.normal(rate_mean, rate_std)

        if rate > 1:
            logging.warning(
                f"{genome}: The pulled dropout rate ({rate:.3f}) based on the given rate_mean ({rate_mean}) and rate_std ({rate_std}) was larger than 1.0 (100%). Pulled rate was ignored, because no sample with all dropped amplicons can exist.")
            continue

        if rate < 0:
            logging.warning(
                f"{genome}: The pulled dropout rate ({rate:.3f}) based on the given rate_mean ({rate_mean}) and rate_std ({rate_std}) was below 0.0. Pulled rate was ignored.")
            continue

        if curr_drop_rate > rate:
            logging.warning(
                f"{genome}: The current dropout rate is: {curr_drop_rate:.2f}. This is larger than the pulled rate ({rate:.3f}) based on the given rate_mean ({rate_mean}) and rate_std ({rate_std}). No extra amplicons will be dropped.")
            continue

        n_to_drop = int((rate * max_df.shape[0]) - n_curr_dropped_amps)

        non_dropped = max_df[~dropped_mask].copy()

        non_dropped["inv_reads"] = 1 / non_dropped["n_reads"]
        non_dropped["drop_prob"] = non_dropped["inv_reads"] / non_dropped["inv_reads"].sum()
        indeces_to_drop.extend(rng.choice(
            a=non_dropped.index,
            p=non_dropped["drop_prob"],
            size=n_to_drop,
            replace=False,
            shuffle=False
        ))

    if len(indeces_to_drop) == 0:
        return amplicon_df
    amplicon_df.loc[indeces_to_drop, "n_reads"] = 0
    return amplicon_df


def add_persistent_mutation_count(amplicon_df: pd.DataFrame, snv_folder: str) -> pd.DataFrame:
    SNV_files = os.listdir(snv_folder)
    for file in SNV_files:
        snv_df = pd.read_csv(os.path.join(snv_folder, file), index_col=False, delimiter="\t")
        if snv_df.empty:
            continue

        alt = int(snv_folder[-1])
        alt_num_mask = (amplicon_df["alt_num_left"] == alt) | (amplicon_df["alt_num_right"])
        genome_mask = amplicon_df["ref"] == snv_df["REGION"].unique()[0]
        mask = alt_num_mask & genome_mask
        amps_in_genome = amplicon_df.loc[mask]
        for _, snv in snv_df.iterrows():
            pos = snv["POS"]
            amplicon_df.loc[mask, "SNVs_in_primers"] += (((
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

        alts = amp_dist_df.loc[amp_dist_df["alt_num_left"] == 2, "amplicon_number"].drop_duplicates()
        unique = amp_dist_df["amplicon_number"].unique()

        normal_hyperparam = 1 / len(unique)
        amp_dist_df["hyperparameter"] = normal_hyperparam
        # four original+alt primer combos
        amp_dist_df.loc[amp_dist_df["amplicon_number"].isin(alts), "hyperparamter"] = normal_hyperparam / 4
    return amp_dist_df
