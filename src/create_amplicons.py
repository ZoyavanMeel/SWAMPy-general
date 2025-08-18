import subprocess
from os.path import join, basename
from io import StringIO
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import logging

import helpers as hp


def align_primers(genome_filename_short: str, indices_folder: str, primers_files: str, verbose: bool = False):

    sp = subprocess.run(
        ["bwa", "mem", "-k", "5", "-L", "1000", "-T", "16", join(indices_folder, genome_filename_short), primers_files],
        # ["bowtie2", "-x", join(indices_folder, genome_filename_short), "-U", primers_files],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    err = sp.stderr.decode()
    for e in ["[e::", "error", "err", "fail"]:
        if e in err.lower():
            err_str = f"BWA error: {err}"
            logging.error(err_str)
            exit(err_str)
    # if "(err)" in err.lower() or "error" in err.lower():
    #     err_str = f"Bowtie2 error: {err}"
    #     logging.error(err_str)
    #     exit(err_str)

    alignment = StringIO(sp.stdout.decode())

    # read alignment data as a dataframe
    df = pd.read_csv(alignment, sep="\t", skiprows=[0, 1], header=None, names=[i for i in range(19)])
    df = pd.DataFrame(df[[0, 2, 3, 9]])
    df = df.rename(columns={0: "name", 2: "ref", 3: "start", 9: "seq"})

    # split the column "name" to extract useful data
    df["seq_len"] = df["seq"].apply(len)

    name_df = hp.process_amplicon_names(df["name"])
    df["amplicon_number"] = name_df["amp_num"]
    df["alt_num"] = name_df["alt_num"]
    df["handedness"] = name_df["handedness"]
    del name_df

    # remove rows where the alignment mismatched
    drop_rows = []
    for r in df.itertuples():
        if r.ref == "*":
            if verbose:
                logging.info(f"Dropping amplicon {r.amplicon_number}, couldn't find a match for the primer {r.name}")
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
    # Z: filepaths are now: /genome_amplicon_\d+_\d+_\d+\.fasta
    df["amplicon_filepath"] = genome_filename_short + "_amplicon_" + df["amplicon_number"].astype(str)
    df["amplicon_filepath"] += "_" + df["alt_num_left"].astype(str)
    df["amplicon_filepath"] += "_" + df["alt_num_right"].astype(str) + ".fasta"
    return df


def write_amplicon(df: pd.DataFrame, reference: SeqRecord, genome_filename_short: str, amplicons_folder: str, verbose: bool = False):

    fasta_entries = []
    reference_string = str(reference.seq)
    for _, row in df.iterrows():
        amplicon_number = row["amplicon_number"]
        lalt, ralt = row["alt_num_left"], row["alt_num_right"]

        amplicon = reference_string[row["left"] - 1: row["right"] + row["right_primer_length"] - 1]

        header = f">{reference.id}_amplicon_{amplicon_number}_{lalt}_{ralt}"
        fasta_entry = f"{header}\n{amplicon}"
        fasta_entries.append(fasta_entry)

    # Z: write all at once to a *single* FASTA file
    with open(join(amplicons_folder, f"all_amplicons_{genome_filename_short}.fasta"), "w") as f:
        f.write("\n".join(fasta_entries))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Create amplicons for a genome using a primer set.")
    parser.add_argument("--genome_path", "-g", help="Path to the genome of interest.")
    parser.add_argument("--amplicons_folder", "-am", help="Folder where the output amplicons will go.")
    parser.add_argument("--indices_folder", "-i", help="Folder where BWA indices are created and stored.")
    parser.add_argument("--primers_file", "-p", help="Path to fastq file of primers. Default ARTIC V1 primers.")
    parser.add_argument("--verbose", help="Verbose mode.")

    args = parser.parse_args()
    genome_filename_short = ".".join(basename(args.genome_path).split(".")[:-1])
    reference = SeqIO.read(args.genome_path, format="fasta")

    hp.build_index(args.genome_path, join(args.indices_folder, genome_filename_short))
    df = align_primers(genome_filename_short, args.indices_folder, args.primers_file, args.verbose)
    write_amplicon(df, reference, genome_filename_short, args.amplicons_folder, verbose=args.verbose)
