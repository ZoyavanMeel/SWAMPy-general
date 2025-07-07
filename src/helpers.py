import os
import logging
import subprocess

import pandas as pd
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SeqRecord


def build_index(genome_path: str, index_base: str, mkdir: bool = False):
    """
    Function to build a Bowtie2 index base.
    - `genome_path`: Filepath to reference FASTA to make the index of.
    - `index_base`: Location+prefix for the index base.
    - `mkdir`: Whether to make the output folder for the index base if it does not yet exist.
               Equivalent to running `mkdir -p` for the index base directory
    Returns: `stdout`, `stderr`
    """
    if mkdir:
        idx_dir = os.path.join(os.path.split(index_base)[:-1])
        subprocess.run(["mkdir", "-p", idx_dir])

    sp = subprocess.run(
        ["bowtie2-build", genome_path, index_base],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    err = sp.stderr.decode()
    # Bowtie2 stderr is just what they for everything
    if "(err)" in err.lower() or "error" in err.lower():
        err_str = f"Bowtie2 error: {err}"
        logging.error(err_str)
        exit(err_str)


def bed_2_fastq(genome_file: str, bed_file: str, fastq_file: str) -> None:
    """
    All this function does is take a bed file of primers (tab separated with columns 'genome', 'start', 'end', 'name'(, 'pool', 'sense'))
    and returns a fastq of these primer with dummy quality scores.
    - `genome_file`: Filepath to FASTA you want to convert
    - `bed_file`: Primer BED-file you want the sequences of
    - `fastq_file`: The output file
    """
    primers = pd.read_csv(bed_file, sep="\t", names=['genome', 'start', 'end', 'name'], usecols=[i for i in range(4)])
    genome = SeqIO.read(genome_file, format="fasta")

    with open(fastq_file, "w") as fh:
        for _, r in primers.iterrows():
            start, end = int(r["start"]), int(r["end"])
            if "_RIGHT" in r["name"]:
                primer = SeqRecord(genome.seq[start:end].reverse_complement())
            else:
                primer = SeqRecord(genome.seq[start:end])
            primer.description = ""
            primer.id = r["name"]
            primer.letter_annotations["solexa_quality"] = [40] * len(primer)
            fh.write(primer.format("fastq"))


def process_amplicon_names(names: pd.Series) -> pd.DataFrame:
    name_df = names.str.split("_", expand=True)
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
        name_df["alt_num"] = 1
    if len(name_df.columns) == 4:
        name_df.columns = ["prefix", "amp_num", "handedness", "alt_num"]
        name_df["alt_num"] = name_df["alt_num"].astype(int)
    name_df["amp_num"] = name_df["amp_num"].astype(int)
    return name_df


if __name__ == "__main__":
    bed_2_fastq("../ref/MN908947.3.fasta", "../primer_sets/artic_v3_all_alt.bed", "../test.fastq")
