import os
import pandas as pd
import numpy as np

from itertools import product

from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def rename_artic_primer(name: str) -> str:
    lst = name.split("_")
    lst[3], lst[2] = lst[2], lst[3]
    return "_".join(lst)


def read_artic_v3(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, delimiter="\t", header=None,
        names=["Chromosome", "Start", "End", "Primer_name", "Score", "Strand", "Seq"]
    )
    df["Primer_name"] = df["Primer_name"].apply(rename_artic_primer)
    return df


def get_alt_amplicon_combos(all_amplicons: pd.Series) -> pd.Series:
    df = all_amplicons.str.split("_", expand=True)

    cols = df.columns.to_list()
    cols[-2], cols[-1] = "Amp_num", "Alt_num"
    df.columns = cols

    df["Alt_num"] = pd.to_numeric(df["Alt_num"])
    df["Amp_num"] = pd.to_numeric(df["Amp_num"])

    df = df[df["Alt_num"] > 1]  # only amplicons with alts
    df = df.loc[df.groupby("Amp_num")["Alt_num"].idxmax()]  # only the highest alternate number

    # ok so this looks really bad, but this never exceeds ~100 iterations in practice
    combos = []
    for i, amp in df.iterrows():
        prefix = "_".join(amp.values[:-2])
        for left_num, right_num in product(range(1, amp["Alt_num"]+1), range(1, amp["Alt_num"]+1), repeat=1):
            if left_num == right_num:  # we already have the "proper" ones
                continue
            combos.append(f"{prefix}_{amp['Amp_num']}_{left_num}/{prefix}_{amp['Amp_num']}_{right_num}")
    return pd.Series(data=[0]*len(combos), index=combos)


def read_primer_bed(path: str, as_artic: bool = False) -> pd.DataFrame:
    if as_artic or "artic_v3_all_alt" in path:
        return read_artic_v3(path)
    cols = ["Chromosome", "Start", "End", "Primer_name", "Feature_overlap", "Strand"]
    return pd.read_csv(path, names=cols, header=None, delimiter="\t")


def get_alt_combos_snv_counts(left: pd.DataFrame, right: pd.DataFrame, primer_bed_path) -> pd.DataFrame:
    all_amplicons = read_primer_bed(primer_bed_path)["Primer_name"].replace(
        r"(_LEFT|_RIGHT)$", "", regex=True).unique()  # .unique() becomes a np.array bruh
    alt_combos = get_alt_amplicon_combos(pd.Series(all_amplicons))
    alt_combos = alt_combos.index.str.split("/", expand=True).to_frame().reset_index(drop=True)
    alt_combos.columns = ["Left", "Right"]

    SNVs_left = left.loc[alt_combos["Left"]].reset_index(names="Primer_name")
    SNVs_left["Primer_name"] += "/"
    SNVs_right = right.loc[alt_combos["Right"]].reset_index(names="Primer_name")

    return (SNVs_left + SNVs_right).set_index("Primer_name")


def rename_summary_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.copy()
    idx: pd.DataFrame = idx.str.split("_", expand=True).to_frame()
    idx.columns = ["amp_num", "alt_num_left", "alt_num_right"]

    idx["prefix"] = "nCoV-2019_" + idx["amp_num"]

    idx["suffix"] = ""
    norm_amps = idx["alt_num_left"] == idx["alt_num_right"]
    idx.loc[norm_amps, "suffix"] = idx.loc[norm_amps, "alt_num_left"]

    alt_df = idx[~norm_amps]
    idx.loc[~norm_amps, "suffix"] = alt_df["alt_num_left"] + "/" + alt_df["prefix"] + "_" + alt_df["alt_num_right"]

    idx["index"] = idx["prefix"] + "_" + idx["suffix"]
    df["index"] = idx["index"].to_list()
    return df.set_index("index", drop=True)


def build_summary_df(dir_path: str) -> None:
    count_dict, snv_dict = {}, {}
    for acc in os.listdir(dir_path):
        df = pd.read_csv(os.path.join(dir_path, acc, f"{acc}_amplicon_abundances_summary.tsv"), sep="\t", index_col=0)
        groupby = df.groupby(["amplicon_number", "alt_num_left", "alt_num_right"])

        df.set_index(keys=["amplicon_number", "alt_num_left", "alt_num_right"], inplace=True)
        df["is_max"] = df["props"] == groupby["props"].transform('max')

        df["READS"] = 0
        df.loc[df["is_max"], "READS"] = groupby["n_reads"].sum()

        count_dict[acc] = df.loc[df["is_max"], "READS"]
        snv_dict[acc] = df.loc[df["is_max"], "SNVs_in_primers"]

    count_df = pd.DataFrame().from_dict(count_dict)
    snv_df = pd.DataFrame().from_dict(snv_dict)

    count_df.fillna(0, inplace=True)

    count_df["index"] = ['_'.join(map(str, i)) for i in count_df.index.tolist()]
    snv_df["index"] = ['_'.join(map(str, i)) for i in snv_df.index.tolist()]

    count_df.set_index("index", drop=True, inplace=True)
    snv_df.set_index("index", drop=True, inplace=True)

    count_df.to_csv("supplementary_files/amplicon_counts_nCOV19_clinical_simulated_summary_1_dataset.csv")
    snv_df.T.to_csv("supplementary_files/nCOV19_clinical_simulated_summary_1.csv")


def load(which: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    count_df = pd.read_csv(
        f"supplementary_files/amplicon_counts_nCOV19_clinical_{which}_1_dataset.csv", index_col=0
    )
    abundance_arr = np.where(
        count_df.median(axis=0) != 0,                   # if median(sample) != 0
        count_df.div(count_df.median(axis=0), axis=1),  # then simply calculate the normalization
        1                                               # else relative abundance = 1
    )
    abundance_df = pd.DataFrame(data=abundance_arr, index=count_df.index, columns=count_df.columns)

    snv_df = pd.read_csv(f"supplementary_files/nCOV19_clinical_{which}_1.csv", index_col=0)
    snv_df = snv_df.fillna(value=0)
    snv_df = snv_df.T

    # fix normalization weirdness
    abundance_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if which == "simulated_summary":
        snv_df = snv_df[abundance_df.columns]
        snv_df = rename_summary_index(snv_df)
        abundance_df = rename_summary_index(abundance_df)
        return abundance_df, snv_df

    left = snv_df.filter(regex="_LEFT", axis=0).copy()
    left.index = left.index.str.strip("_LEFT")

    right = snv_df.filter(regex="_RIGHT", axis=0).copy()
    right.index = right.index.str.strip("_RIGHT")

    lr_df = left + right

    alt_df = get_alt_combos_snv_counts(left, right, "primer_sets/artic_v3_all_alt.bed")
    lr_df = pd.concat([lr_df, alt_df])

    # match columns (+ order)
    lr_df = lr_df[abundance_df.columns]

    return abundance_df, lr_df


def compare() -> None:
    real_abu_df, real_snv_df = load("real")
    sim_abu_df, sim_snv_df = load("simulated")
    sum_abu_df, sum_snv_df = load("simulated_summary")
    tol_abu_df, tol_snv_df = load("tol")

    # other way around is empty, snv_df is same
    diff = set(sim_abu_df.index).difference(sum_abu_df.index)
    for amp in diff:
        sum_abu_df.loc[amp] = [np.nan] * len(sum_abu_df.columns)
        sum_snv_df.loc[amp] = [np.nan] * len(sum_snv_df.columns)

    diff = set(sim_abu_df.index).difference(tol_abu_df.index)
    for amp in diff:
        tol_abu_df.loc[amp] = [np.nan] * len(tol_abu_df.columns)
        tol_snv_df.loc[amp] = [np.nan] * len(tol_snv_df.columns)

    print(
        pd.concat(
            [
                (real_snv_df.fillna(0) != 0).sum(axis=1).fillna(0).astype(int),
                (sum_snv_df.fillna(0) != 0).sum(axis=1).fillna(0).astype(int),
                (sim_snv_df.fillna(0) != 0).sum(axis=1).fillna(0).astype(int),
            ], axis=1, ignore_index=False
        )
    )
    quit()

    long_sum_abu = sum_abu_df.melt(var_name="accession", value_name="abundance", ignore_index=False)
    long_sum_abu["set"] = "summary"
    long_sum_abu = long_sum_abu.sort_index(
        key=lambda idx: idx.str.extract(r'_(\d+)_(\d+)').astype(int).apply(tuple, axis=1)
    ).reset_index(names="amplicon")
    long_sim_abu = sim_abu_df.melt(var_name="accession", value_name="abundance", ignore_index=False)
    long_sim_abu["set"] = "simulated"
    long_sim_abu = long_sim_abu.sort_index(
        key=lambda idx: idx.str.extract(r'_(\d+)_(\d+)').astype(int).apply(tuple, axis=1)
    ).reset_index(names="amplicon")
    long_real_abu = real_abu_df.melt(var_name="accession", value_name="abundance", ignore_index=False)
    long_real_abu["set"] = "real"
    long_real_abu = long_real_abu.sort_index(
        key=lambda idx: idx.str.extract(r'_(\d+)_(\d+)').astype(int).apply(tuple, axis=1)
    ).reset_index(names="amplicon")
    long_tol_abu = tol_abu_df.melt(var_name="accession", value_name="abundance", ignore_index=False)
    long_tol_abu["set"] = "tol"
    long_tol_abu = long_tol_abu.sort_index(
        key=lambda idx: idx.str.extract(r'_(\d+)_(\d+)').astype(int).apply(tuple, axis=1)
    ).reset_index(names="amplicon")

    # print(pd.concat([long_sum_abu, long_sim_abu, long_real_abu]))
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(
        data=pd.concat([long_sum_abu, long_tol_abu, long_sim_abu], ignore_index=True),
        ax=ax, x="amplicon", y="abundance", hue="set", fliersize=0,
        notch=True, medianprops={"color": "r", "linewidth": 2}
    )
    plt.xticks(rotation=90)
    plt.ylim(0, 20)
    low, up = 0, 130
    plt.xlim(low-0.5, up+.5)
    plt.tight_layout()
    plt.show()

    # plot(real_abu_df, real_snv_df)
    # plot(sim_abu_df, sim_snv_df)
    # plot(sum_abu_df, sum_snv_df)


def plot(abu_df: pd.DataFrame, snv_df: pd.DataFrame) -> None:
    zero = (snv_df == 0).to_numpy().flatten()
    one = (snv_df == 1).to_numpy().flatten()
    two = (snv_df == 2).to_numpy().flatten()
    drop: np.ndarray = (abu_df == 0).to_numpy().flatten()

    zero_snvs = abu_df.to_numpy().flatten()[~drop & zero]
    zero_snvs = zero_snvs[~np.isnan(zero_snvs)]
    one_snvs = abu_df.to_numpy().flatten()[~drop & one]
    one_snvs = one_snvs[~np.isnan(one_snvs)]
    two_snvs = abu_df.to_numpy().flatten()[~drop & two]
    two_snvs = two_snvs[~np.isnan(two_snvs)]

    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(
        [zero_snvs, one_snvs, two_snvs], fliersize=0, ax=ax, color="b",
        notch=True, medianprops={"color": "r", "linewidth": 2}
    )
    plt.xticks(
        ticks=[0, 1, 2],
        labels=[f"0 SNVs (n={len(zero_snvs)})", f"1 SNVs (n={len(one_snvs)})", f"2 SNVs (n={len(two_snvs)})"],
        rotation=45
    )
    plt.ylabel("Relative amplicon abundance")
    plt.ylim(0, 7)
    plt.yticks(np.arange(0, 7.5, 0.5))

    plt.grid(True, "both", "y")
    plt.tight_layout()
    plt.show()

    no_snv_mask = snv_df == 0
    dropped_mask = abu_df == 0

    flat_no_snv_mask = no_snv_mask.to_numpy().flatten()
    flat_dropped_mask = dropped_mask.to_numpy().flatten()

    yes_snvs = abu_df.to_numpy().flatten()[~flat_dropped_mask & ~flat_no_snv_mask]
    yes_snvs = yes_snvs[~np.isnan(yes_snvs)]
    no_snvs = abu_df.to_numpy().flatten()[~flat_dropped_mask & flat_no_snv_mask]
    no_snvs = no_snvs[~np.isnan(no_snvs)]

    stat, p_less = mannwhitneyu(yes_snvs, no_snvs, alternative="less")
    print(p_less)

    a = (dropped_mask & ~no_snv_mask).sum().sum()
    b = (dropped_mask & no_snv_mask).sum().sum()
    c = (~dropped_mask & ~no_snv_mask).sum().sum()
    d = (~dropped_mask & no_snv_mask).sum().sum()

    contingency = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(contingency, alternative="greater")
    conf_int_rr = Table2x2(contingency).riskratio_confint()
    print(p_value)
    print(conf_int_rr)


if __name__ == "__main__":
    # build_summary_df("simulation_output/nCOV19_clinical_simulated_1")
    compare()
