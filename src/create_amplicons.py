import subprocess
from io import StringIO
import os
import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import logging

import helpers as hp


def get_alignment_df_and_call_SNVs(
    ref_path: str, genome_filename_short: str, indices_folder: str,
    primer_fastq: str, primer_bed: str, temp_folder: str, score_thresh: int,
    verbose: bool = False, no_align: str = "raise"
):
    """Align primers to lineages"""

    pool1_path, pool2_path = hp.split_primers_for_snv_pools(primer_bed, primer_fastq, temp_folder)

    df_one = call_SNVs(ref_path, genome_filename_short, indices_folder,
                       pool1_path, os.path.join(temp_folder, "SNV1"), score_thresh)
    df_two = call_SNVs(ref_path, genome_filename_short, indices_folder,
                       pool2_path, os.path.join(temp_folder, "SNV2"), score_thresh)

    df = pd.concat([df_one, df_two])

    # split the column "name" to extract useful data
    df["seq_len"] = df["seq"].apply(len)

    name_df = hp.process_amplicon_names(df["name"])
    df["amplicon_number"] = name_df["amp_num"]
    df["alt_num"] = name_df["alt_num"]
    df["handedness"] = name_df["handedness"]
    del name_df

    # remove rows where the primer didn't align
    keep = df["ref"] != "*"
    if verbose:
        for name in df[~keep]["name"].to_list():
            logging.info(f"Couldn't find a match for the primer: {name}.\tDropping corresponding amplicon.")
    df = df[keep]
    if df.empty and no_align == "raise":
        raise ValueError(f"None of the primers aligned to the given genome ({ref_path})! Can't simulate any reads.")
    if df.empty and no_align == "warn":
        logging.warn(f"None of the primers aligned to the given genome ({ref_path})! Can't simulate any reads.")
        return

    # process alignment score tag (after dropping rows because of NaN scores)
    df["align_score"] = df["align_score"].str.split(":").str[-1].astype(int)

    # Z: Merging with `suffixes` and on different columns makes this easier
    # Z: Does not merge on `is_alt` anymore to get every combination of fw/rv primers with alts (all biologically viable)
    # inner join the dataframe with itself, to get the pairs of primers and their start/end positions
    df = pd.merge(
        df.loc[df["handedness"] == "LEFT"],
        df.loc[df["handedness"] == "RIGHT"],
        on=["ref", "amplicon_number"],
        suffixes=["_left", "_right"]
    )
    if df.empty and no_align == "raise":
        raise ValueError(
            f"No amplicons where both primers aligned to the given genome ({ref_path})! Can't simulate any reads.")
    if df.empty and no_align == "warn":
        logging.warning(
            f"No amplicons where both primers aligned to the given genome ({ref_path})! Can't simulate any reads.")
        return

    # # Pick one "best" amplicon for those with alternates
    df["align_score"] = df["align_score_left"] + df["align_score_right"]
    # df = df.sort_values("align_score").drop_duplicates("amplicon_number", keep='last').sort_index()

    # rename the columns to more understandable names
    df = df[[
        "ref", "amplicon_number", "alt_num_left",
        "start_left", "seq_left", "seq_len_left", "alt_num_right",
        "start_right", "seq_right", "seq_len_right", "align_score"
    ]]

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


def call_SNVs(ref_path, genome_filename_short, indices_folder, primer_fastq, snv_folder, score_thresh):
    bwa_alignment = subprocess.run(
        [
            "bwa", "mem", "-k", "5", "-L", "1000", "-T", str(score_thresh),
            os.path.join(indices_folder, genome_filename_short), primer_fastq
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    hp.check_stderr(bwa_alignment.stderr.decode(), "BWA")

    # Alignment is needed for futher processing, but we don't need
    # to make that data persistent if we call SNVs now.
    sam_view_sam = subprocess.run(
        # remove sam header
        ["samtools", "view", "-F", "0x800", "-"],
        input=bwa_alignment.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    alignment = sam_view_sam.stdout.decode()

    sam_view_bam = subprocess.run(
        ["samtools", "view", "-b", "-"],
        input=bwa_alignment.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    sam_sort = subprocess.run(
        ["samtools", "sort", "-"],
        input=sam_view_bam.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    sam_mpile = subprocess.run(
        ["samtools", "mpileup", "-aa", "-A", "-d", "600000", "-B", "-Q", "0", "--excl-flags", "0x804", "-"],
        input=sam_sort.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if len(sam_mpile.stdout) == 0:
        logging.error("Something went wrong in primer-to-lineage alignment!")
        exit(1)

    os.makedirs(snv_folder, exist_ok=True)
    ivar_variants = subprocess.run(
        ["ivar", "variants", "-p", os.path.join(snv_folder, genome_filename_short + ".tsv"),
            "-t", "0.99", "-r", ref_path],
        input=sam_mpile.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    hp.check_stderr(ivar_variants.stderr.decode(), "iVar")
    hp.filter_ambigious_nucleotides(snv_folder)

    # read alignment data as a dataframe
    df = pd.read_csv(
        StringIO(alignment), sep="\t", header=None, names=[i for i in range(19)]
    )
    df = pd.DataFrame(df[[0, 2, 3, 9, 13]])
    df = df.rename(columns={0: "name", 2: "ref", 3: "start", 9: "seq", 13: "align_score"})
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
    with open(os.path.join(amplicons_folder, f"all_amplicons_{genome_filename_short}.fasta"), "w") as f:
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
    genome_filename_short = ".".join(os.path.basename(args.genome_path).split(".")[:-1])
    reference = SeqIO.read(args.genome_path, format="fasta")

    hp.build_index(args.genome_path, os.path.join(args.indices_folder, genome_filename_short))
    df = get_alignment_df_and_call_SNVs(genome_filename_short, args.indices_folder, args.primers_file, args.verbose)
    write_amplicon(df, reference, genome_filename_short, args.amplicons_folder, verbose=args.verbose)
