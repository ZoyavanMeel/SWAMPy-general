import subprocess
from os.path import join, basename
from io import StringIO
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import logging

import helpers as hp


def align_primers(genome_filename_short: str, indices_folder: str, primers_files: str, verbose: bool = False):

    # run bowtie2 aligner
    # alignment = subprocess.run(
    #     ["bowtie2", "-x", join(indices_folder, genome_filename_short), "-U", primers_files],
    #     capture_output=True
    # )

    sp = subprocess.run(
        ["bowtie2", "-x", join(indices_folder, genome_filename_short), "-U", primers_files],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    err = sp.stderr.decode()
    if "(err)" in err.lower() or "error" in err.lower():
        err_str = f"Bowtie2 error: {err}"
        logging.error(err_str)
        exit(err_str)

    alignment = StringIO(sp.stdout.decode("UTF-8"))

    # read alignment data as a dataframe
    df = pd.read_csv(alignment, sep="\t", skiprows=[0, 1, 2], header=None, names=[i for i in range(19)])
    df = pd.DataFrame(df[[0, 2, 3, 9]])
    df = df.rename(columns={0: "name", 2: "ref", 3: "start", 9: "seq"})

    # split the column "name" to extract useful data
    df["seq_len"] = df["seq"].apply(len)
    df["is_alt"] = False

    name_df = df["name"].str.split("_", expand=True)
    if len(name_df.columns) < 3 or len(name_df.columns) > 4:
        print(name_df.columns)
        print(len(name_df.columns))
        print(name_df.head())
        err_str = """Primer names don't follow specification.
Must be: '<prefix>_<amplicon number>_<LEFT|RIGHT>' or '<prefix>_<amplicon number>_<LEFT|RIGHT>_<alternate number>'.
No underscores or forward slashes allowed in <prefix>."""
        logging.error(err_str)
        exit(err_str)

    if len(name_df.columns) == 3:
        name_df.columns = ["prefix", "amp_num", "handedness"]
        df["alt_num"] = 1
    if len(name_df.columns) == 4:
        name_df.columns = ["prefix", "amp_num", "handedness", "alt_num"]
        df["alt_num"] = name_df["alt_num"].astype(int)

    df["amplicon_number"] = name_df["amp_num"].astype(int)
    df["handedness"] = name_df["handedness"]
    del name_df

    # remove rows where the alignment mismatched
    drop_rows = []
    for r in df.itertuples():
        if r.ref == "*":
            if verbose:
                logging.info(f"Dropping amplicon {r.amplicon_number}, couldn't find a match for the primer {r.seq}")
            drop_rows.append(r.Index)

    df.drop(drop_rows, inplace=True)

    # Z: Merging with `suffixes` and on different columns makes this easier
    # Z: Does not merge on `is_alt` anymore to get every combination of fw/rv primers with alts (all biologically viable)
    # inner join the dataframe with itself, to get the pairs of primers and their start/end positions
    df = pd.merge(
        df.loc[df["handedness"] == "LEFT"],
        df.loc[df["handedness"] == "RIGHT"],
        on=["ref", "amplicon_number"],
        suffixes=["_left", "_right"]
    )

    # rename the columns to more understandable names
    df = pd.DataFrame(
        df[["ref", "amplicon_number", "alt_num_left",
            "start_left", "seq_left", "seq_len_left", "alt_num_right",
            "start_right", "seq_right", "seq_len_right"]]
    )

    df = df.rename(columns={
        "start_left": "left",
        "seq_left": "left_primer",
        "seq_len_left": "left_primer_length",
        "start_right": "right",
        "seq_right": "right_primer",
        "seq_len_right": "right_primer_length"
    })

    # ampNum_altNumLeft_altNumRight
    # Z: filepaths are now: /genome_amplicon_\d+_\d+_\d+\.fasta/
    df["amplicon_filepath"] = genome_filename_short + "_amplicon_" + \
        df["amplicon_number"].map(str) + "_" + df["alt_num_left"].map(str) + \
        "_" + df["alt_num_right"].map(str) + ".fasta"

    if verbose:
        logging.info("First 5 rows: ")
        logging.info(df.head())

    return df


def write_amplicon(df: pd.DataFrame, reference: SeqRecord, genome_filename_short: str, amplicons_folder: str, verbose: bool = False):

    for _, row in df.iterrows():
        amplicon_number = row["amplicon_number"]
        lalt, ralt = row["alt_num_left"], row["alt_num_right"]
        reference_string = str(reference.seq)
        amplicon = reference_string[row["left"] - 1: row["right"] + row["right_primer_length"] - 1]

        if verbose:
            logging.info(f">{reference.id}_amplicon_{amplicon_number}_{lalt}_{ralt}")
            logging.info("length: " + str(row.right - row.left))
            logging.info(amplicon + "\n")
            logging.info(row["left_primer"] + "-" * (row["right"] - row["left"] -
                         row["left_primer_length"]) + row["right_primer"] + "\n")

        with open(f"{amplicons_folder}/{genome_filename_short}_amplicon_{amplicon_number}_{lalt}_{ralt}.fasta", "w") as f:
            f.write(f">{reference.id}_amplicon_{amplicon_number}_{lalt}_{ralt}\n")
            f.write(amplicon + "\n\n")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Create amplicons for a genome using a primer set.")
    parser.add_argument("--genome_path", "-g", help="Path to the genome of interest.")
    parser.add_argument("--amplicons_folder", "-am", help="Folder where the output amplicons will go.")
    parser.add_argument("--indices_folder", "-i", help="Folder where bowtie2 indices are created and stored.")
    parser.add_argument("--primers_file", "-p", help="Path to fastq file of primers. Default ARTIC V1 primers.")
    parser.add_argument("--verbose", help="Verbose mode.")

    args = parser.parse_args()
    genome_filename_short = ".".join(basename(args.genome_path).split(".")[:-1])
    reference = SeqIO.read(args.genome_path, format="fasta")

    hp.build_index(args.genome_path, join(args.indices_folder, genome_filename_short))
    df = align_primers(genome_filename_short, args.indices_folder, args.primers_file, args.verbose)
    write_amplicon(df, reference, genome_filename_short, args.amplicons_folder, verbose=args.verbose)
