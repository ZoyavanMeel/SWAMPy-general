import os
import pandas as pd
import argparse


def build_summary_df(sim_output_dir: str, output_dir) -> None:
    count_dict, snv_dict = {}, {}
    for acc in os.listdir(sim_output_dir):
        df = pd.read_csv(os.path.join(sim_output_dir, acc,
                         f"{acc}_amplicon_abundances_summary.tsv"), sep="\t", index_col=0)
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

    path = os.path.join(*os.path.split(output_dir)[:-1])
    file = os.path.split(output_dir)[-1]

    count_df.to_csv(os.path.join(path, f"amplicon_counts_{file}_dataset.csv"))
    snv_df.T.to_csv(os.path.join(path, file+".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    build_summary_df(args.sim_output_dir, args.output_dir)
