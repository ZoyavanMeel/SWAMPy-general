import os
from functools import partial
import numpy as np
import random
from Bio import SeqIO
import pandas as pd
import subprocess
from io import StringIO
import re
import logging

import helpers as hp

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

ISSUE_6_BUG_CODE = -1


def alts(ref: str, type: str, len: int = 0) -> str:
    # Produce an alternative allele for a given error.
    nucs = ["A", "C", "G", "T"]
    if type == "SUBS":
        nucs.remove(ref)
        return random.choice(nucs)
    elif type == "DEL":
        return ref[0]
    else:  # insertion
        insert = random.choices(nucs, k=len)
        return ref + "".join(insert)


def amplicon_lookup(primer_df: pd.DataFrame, position: int, recurrent: bool) -> list[str]:
    # Find amplicons corresponding to a given position

    mask = (position >= primer_df["Start_left"]) & (position <= primer_df["End_right"])
    corresponding_amplicons = primer_df.loc[mask, "amplicon_number"].tolist()

    if len(corresponding_amplicons) > 0 and not recurrent:
        return random.sample(corresponding_amplicons, k=1)
    return corresponding_amplicons


def no_del_in_disallowed(errors: pd.DataFrame, disallowed: np.ndarray) -> pd.DataFrame:
    dels = errors["errortype"] == "DEL"
    deletions = errors[dels].copy()

    if deletions.empty:
        return errors

    deletions["start"] = deletions["pos"]
    deletions["end"] = deletions["pos"] + deletions["length"]

    # For each deletion, check if any disallowed position is inside [start, end]
    overlaps = (
        deletions["start"].to_numpy().reshape(-1, 1) <= disallowed
    ) & (
        deletions["end"].to_numpy().reshape(-1, 1) >= disallowed
    )
    deletions["overlaps"] = overlaps.any(axis=1)

    errors_clean = pd.concat([
        deletions[~deletions["overlaps"]],
        errors[~dels],
    ])

    return errors_clean.drop(columns=["start", "end", "overlaps"])


def get_sequence_params(row: pd.Series, REFSEQ, VAF_DICT: dict, R_VAF_DICT: dict, genome_abundances: dict, rng: np.random.Generator) -> pd.Series:
    ref = REFSEQ.seq[row["pos"]] if row["errortype"] != "DEL" else REFSEQ.seq[row["pos"]-1:row["pos"]+row["length"]]
    alt = alts(ref, row["errortype"], row["length"])

    if not row["recurrent"]:
        vaf = rng.dirichlet(VAF_DICT[row["errortype"]], size=None)[0]
        genome = random.choices(
            population=list(genome_abundances.keys()),
            weights=list(genome_abundances.values()), k=1)
    else:
        vaf = rng.dirichlet(R_VAF_DICT[row["errortype"]], size=None)[0]
        genome = list(genome_abundances.keys())
    return pd.Series([genome, ref, alt, vaf])


def add_PCR_errors(
    df_amplicons: pd.DataFrame, genome_abundances: dict,
    PATHS: dict[str, str], REF_NAME: str, RATES: dict[str, float],
    DEL_LENGTH_GEOMETRIC_PARAMETER: float, INS_MAX_LENGTH: int,
    VAF_DICT: dict[str, float], R_VAF_DICT: dict[str, float],
    DISALLOWED_POSITIONS: set[int], rng: np.random.Generator
):
    REF = SeqIO.read(PATHS["REFERENCE"], format="fasta")

    U_SUBS_COUNT = int(rng.poisson(RATES["U_SUBS_RATE"]*len(REF.seq), 1))  # unique
    U_INS_COUNT = int(rng.poisson(RATES["U_INS_RATE"]*len(REF.seq), 1))  # unique
    U_DEL_COUNT = int(rng.poisson(RATES["U_DEL_RATE"]*len(REF.seq), 1))  # unique
    R_SUBS_COUNT = int(rng.poisson(RATES["R_SUBS_RATE"]*len(REF.seq), 1))  # recurrent
    R_INS_COUNT = int(rng.poisson(RATES["R_INS_RATE"]*len(REF.seq), 1))  # recurrent
    R_DEL_COUNT = int(rng.poisson(RATES["R_DEL_RATE"]*len(REF.seq), 1))  # recurrent
    SUBS_COUNT = U_SUBS_COUNT+R_SUBS_COUNT
    INS_COUNT = U_INS_COUNT+R_INS_COUNT
    DEL_COUNT = U_DEL_COUNT+R_DEL_COUNT

    if SUBS_COUNT+INS_COUNT+DEL_COUNT == 0:
        return "No", "PCR", "ERROR"

    # create a dataframe of errors that we want to introduce
    errors = build_error_df(
        genome_abundances, PATHS, DEL_LENGTH_GEOMETRIC_PARAMETER, INS_MAX_LENGTH,
        VAF_DICT, R_VAF_DICT, DISALLOWED_POSITIONS, REF, U_SUBS_COUNT, U_INS_COUNT,
        U_DEL_COUNT, R_SUBS_COUNT, R_INS_COUNT, R_DEL_COUNT, SUBS_COUNT, INS_COUNT, DEL_COUNT, rng
    )

    # all amplicons to be mutated
    error_amplicons = [item for sublist in errors["amplicons"] for item in sublist]

    # corresponding error index for that amplicon                 # str even though they're lists; that's just pandas
    indices = np.repeat(a=errors.index, repeats=errors["amplicons"].str.len())

    # these 2 lists will be returned and later passed to art_illumina
    amplicons = []
    n_reads = []

    # Z: build index for reference (SWAMPy originally ships the Wuhan index)
    idx_exts = [".amb", ".ann", ".bwt", ".pac", ".sa"]
    # idx_exts = [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]
    if not all([os.path.exists(PATHS["INDEX_BASE"]+ext) for ext in idx_exts]):
        hp.build_index(PATHS["REFERENCE"], PATHS["INDEX_BASE"])

    # align the original amplicon to the reference because there could be real indels in the source genome.
    align_dfs = {
        lineage: align_amps_to_ref(PATHS, f"all_amplicons_{lineage}.fasta")
        for lineage in genome_abundances.keys()
    }

    def amplicon_alignment(row) -> pd.Series:
        """Simple getter for the align_dfs so that I don't have to change too much of the original structure"""
        df = align_dfs[row["ref"]]
        return df[df["amplicon_number"] == str(row["amplicon_number"])].reset_index(drop=True)

    for _, row in df_amplicons.iterrows():
        # Sometimes more than 1 error are introduced to the same amplicon.
        # Find all errors that will be introduced to that amplicon
        # Z: this is *fine* with alternate amplicons, because they will overlap nearly the exact same region
        #    we check later what the alts are doing
        mut_indices = [
            indices[idx]
            for idx, a in enumerate(error_amplicons)
            if a == row["amplicon_number"] and row["ref"] in errors.loc[indices[idx], "genome"]
        ]

        if len(mut_indices) == 0:
            amplicons.append(row["amplicon_filepath"])
            n_reads.append(row["n_reads"])
            continue

        # TODO: handle alternate amplicons
        # rn it only checks on matching amplicon number, which is fine,
        # but it means that `alignment` can be a DataFrame instead of a Series
        # -> needs to be handled accordingly
        alignment = amplicon_alignment(row)

        # if the amplicon contains too many Ns it will not align, skip introducing PCR error to those
        # with Bowtie2: won't align
        # with BWA: soft-clip like crazy
        short_cigar = alignment["CIGAR"][0]
        if short_cigar == "*" or "S" in short_cigar:
            amplicons.append(row["amplicon_filepath"])
            n_reads.append(row["n_reads"])
            continue

        # Start position and the sequence of the alignment(amplicon)
        start_p = alignment["start"][0]-1
        seq = alignment["seq"][0]

        # Record the CIGAR as a long string. i.e. "MMMII" instead of "3M2I"
        CIGAR = get_long_cigar(short_cigar)

        # Create an empty dataframe to hold how many reads each error combination will produce at the end.
        reads_df = pd.DataFrame()

        # For each error that will be introduced to this amplicon,
        # find the indices for slicing wrt to amplicon seq left end.
        seq_pos = []
        for mut_idx in mut_indices:
            # we aim for the error's position wrt reference genome
            aim = errors.loc[mut_idx, "pos"]
            # amplicon slicing index starts from 0 (left end)
            seq_idx = 0
            # reference index starts from the position where amplicon alignment starts.
            ref_idx = start_p

            # check amp (query) and ref (ref) consumption
            for c_idx, c in enumerate(CIGAR):
                if ref_idx == aim:
                    seq_pos.append(seq_idx)
                    break

                if c == "M":
                    seq_idx += 1
                    ref_idx += 1
                elif c == "D":
                    ref_idx += 1
                elif c == "I":
                    seq_idx += 1

                if c_idx == len(CIGAR)-1 and ref_idx == aim:  # final letter
                    seq_pos.append(seq_idx)
                # CIGAR is shorter, all deletions at the end, skip the error
                elif c_idx == len(CIGAR)-1 and ref_idx != aim:
                    logging.warning(
                        f"A PCR error is skipped since the position does not exist in the amplicon {row['amplicon_filepath']}. This is not a significant problem if you see only one of this warning. Otherwise see Extra options and potential bugs section."
                    )
                    seq_pos.append(ISSUE_6_BUG_CODE)  # dummy placeholder (issue #6)

            # How many reads this specific error will have
            mut_reads = rng.binomial(row["n_reads"], errors.loc[mut_idx, "VAF"])

            # if number of reads and/or VAF are small, this can be 0
            if mut_reads != 0:
                # Take that many samples from the total of the imaginary reads of that amplicon
                reads = sorted(random.sample(range(row["n_reads"]), k=mut_reads))
                reads = [str(a) for a in reads]
                reads = [a+"," for a in reads]  # , will be used for grouping

                muts = [str(mut_idx)]*mut_reads  # keep track of the mutation
                muts = [a+"," for a in muts]

                read_df = pd.DataFrame(dict(reads=reads, muts=muts))
                reads_df = pd.concat([reads_df, read_df], ignore_index=True)

        # if there are no errors with a non-zero count, skip introducing errors to that amplicon
        if reads_df.empty:
            amplicons.append(row["amplicon_filepath"])
            n_reads.append(row["n_reads"])
        else:
            # group wrt imaginary reads to see which ones ended up with wich errors
            reads_df = reads_df.groupby("reads").sum()
            reads_df["count"] = 1
            # group by different combinations of errors to count how many read each combination will produce.
            reads_df = reads_df.groupby("muts", as_index=False).sum()
            # remove the , and turn tham into a list
            reads_df["muts"] = reads_df.apply(lambda x: x["muts"].split(",")[:-1], axis=1)

            # create a dataframe of all errors of the amplicon. Contains pos, mut_index, errortype, length, alt
            seq_pos_df = pd.DataFrame(dict(seq_pos=seq_pos, mut_indices=mut_indices))
            seq_pos_df = seq_pos_df.merge(errors[["errortype", "mut_indices", "length", "alt"]], on="mut_indices")
            # skip errors that correspond to deletions (issue #6)
            seq_pos_df = seq_pos_df[seq_pos_df['seq_pos'] != ISSUE_6_BUG_CODE]

            # amplicon's number of reads - total count of all error combination versions is the count of non-mutated (old) version.
            amplicons.append(row["amplicon_filepath"])
            n_reads.append(row["n_reads"] - sum(reads_df["count"]))

            # for all error combination versions of the amplicon
            for idx, pcr_error in enumerate(reads_df.itertuples()):
                # if a specific combination has 0 reads, pass
                if pcr_error.count != 0:
                    # create a final df that contains all the errors in that specific combination
                    final_df = seq_pos_df.loc[seq_pos_df["mut_indices"].isin([int(a) for a in pcr_error["muts"]])]
                    final_df = final_df.sort_values('seq_pos')
                    final_df.reset_index(drop=True, inplace=True)

                    # introduce those errors one by one.
                    new_seq = ""
                    for indx, final in enumerate(final_df.itertuples()):

                        # the part up to the first error is the same
                        if indx == 0:
                            new_seq += seq[0:final.seq_pos]

                        # if an error is substition or indel, take the part up to and excluding the error position
                        # add alternative instead of the ref at error pos.
                        if final.errortype == "SUBS" or final.errortype == "INS":
                            new_seq = new_seq+final.alt
                            # then add the part up to the next error
                            try:
                                new_seq += seq[final.seq_pos+1:final_df.loc[indx+1, "seq_pos"]]
                            # if it is the last error, add all the remaining sequence
                            except KeyError:
                                new_seq += seq[final.seq_pos+1:]

                        elif final.errortype == "DEL":
                            # if it is a deletion add the next section of the sequence but leave out the first n bases of it
                            try:
                                new_seq += seq[final.seq_pos+1:final_df.loc[indx+1, "seq_pos"]][final.length-1:]
                            except KeyError:
                                new_seq += seq[final.seq_pos+1:][final.length-1:]

                    # add the new amplicon to the list
                    new_path = row["amplicon_filepath"][:-6] + "_p" + str(idx+1) + ".fasta"
                    amplicons.append(new_path)
                    n_reads.append(pcr_error.count)

                    # write the fasta file of the new amplicon.
                    # Name all the PCR error combinations as _p1, _p2 and etc.
                    with open(os.path.join(PATHS['AMPLICONS_FOLDER'], new_path), "w") as new_a:
                        new_a.write(f">{new_path[:-6]}\n")
                        new_a.write(new_seq + "\n\n")

    # this is for optional VCF output.
    errors['chr'] = REF_NAME
    errors['qual'] = "."
    errors['filter'] = "."
    errors['id'] = "."
    errors['pos_0'] = errors.apply(lambda x: x.pos if x.errortype != "DEL" else x.pos-1, axis=1)
    errors['pos_1'] = errors.apply(lambda x: x.pos_0+1, axis=1)
    def r_or_u(x): return "R" if x["recurrent"] else "U"
    errors['info'] = errors.apply(lambda x: "VAF=%.5f" % round(x.VAF, 5) + f";REC={r_or_u(x)}", axis=1)
    errors.sort_values("pos", inplace=True)
    vcf_errordf = errors.loc[:, ["chr", "pos_1", "id", "ref", "alt", "qual", "filter", "info"]]
    return amplicons, n_reads, vcf_errordf


def get_long_cigar(short_cigar: str):
    long_cigar = ""
    for idx, cigar in enumerate(re.split("(M|D|I)", short_cigar)[:-1]):
        if idx % 2 == 0:
            prev = cigar[:]
        else:
            long_cigar += cigar*int(prev)
    return long_cigar


def build_error_df(
    genome_abundances: dict, PATHS: dict, DEL_LENGTH_GEOMETRIC_PARAMETER: float,
    INS_MAX_LENGTH: int, VAF_DICT: dict, R_VAF_DICT: dict, DISALLOWED_POSITIONS: set[int],
    REF, U_SUBS_COUNT, U_INS_COUNT, U_DEL_COUNT, R_SUBS_COUNT, R_INS_COUNT, R_DEL_COUNT,
    SUBS_COUNT, INS_COUNT, DEL_COUNT, rng: np.random.Generator
):
    errors = pd.DataFrame(dict(errortype=["SUBS"]*SUBS_COUNT + ["DEL"]*DEL_COUNT + ["INS"]*INS_COUNT))

    # Z: recurrent = True; unique = False
    errors["recurrent"] = [True]*R_SUBS_COUNT + [False]*U_SUBS_COUNT + [True]*R_DEL_COUNT + \
        [False]*U_DEL_COUNT + [True]*R_INS_COUNT + [False]*U_INS_COUNT

    errors["mut_indices"] = errors.index
    errors["length"] = [1]*SUBS_COUNT +\
        list(rng.geometric(p=DEL_LENGTH_GEOMETRIC_PARAMETER, size=DEL_COUNT)) +\
        random.choices(population=list(range(1, INS_MAX_LENGTH+1)), k=INS_COUNT)

    errors["pos"] = random.sample(population=list(range(len(REF.seq))), k=SUBS_COUNT+INS_COUNT+DEL_COUNT)

    # don't allow substitutions in the disallowed positions
    errors.at[0, "errortype"] = "DEL"
    errors = errors[~(errors["errortype"] == "SUBS") | ~(errors["pos"].isin(DISALLOWED_POSITIONS))]

    # don't allow deletions of the disallowed positions
    # Z: added functionality
    if len(DISALLOWED_POSITIONS) > 0:
        errors = no_del_in_disallowed(errors, np.array(list(DISALLOWED_POSITIONS)))

    # Z: loop over all rows only once instead of 4 times
    get_sequence_params_p = partial(
        get_sequence_params, REFSEQ=REF, VAF_DICT=VAF_DICT,
        R_VAF_DICT=R_VAF_DICT, genome_abundances=genome_abundances, rng=rng
    )
    errors[["genome", "ref", "alt", "VAF"]] = errors.apply(get_sequence_params_p, axis=1)
    errors = errors.loc[errors['VAF'] != 0]

    # Z: don't read the BED-file on each interation, just load it once
    # Kept out of `get_sequence_params` for readability
    primer_df = hp.read_primer_bed(PATHS["PRIMER_BED"])
    errors["amplicons"] = errors.apply(lambda x: amplicon_lookup(primer_df, x["pos"], x["recurrent"]), axis=1)
    return errors


def align_amps_to_ref(PATHS, amp_path: str):
    amp_path = re.sub(r"[/,\|&]", r"&", amp_path).replace(" ", "_")
    alignment = subprocess.run(
        ["bwa", "mem", "-L", "1000000", PATHS['INDEX_BASE'], f"{PATHS['AMPLICONS_FOLDER']}/{amp_path}"],
        # ["bowtie2", "-x", PATHS['INDEX_BASE'], "-f", f"{PATHS['AMPLICONS_FOLDER']}/{amp_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = alignment.stdout.decode(), alignment.stderr.decode()
    for e in ["[e::", "error", "err", "fail"]:
        if e in err.lower():
            err_str = f"BWA error: {err}"
            logging.error(err_str)
            exit(err_str)

    # filter out supplementary alignments
    samview = subprocess.run(
        ["samtools", "view", "-F", "0x800", "-h", "-"],
        input=alignment.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = samview.stdout.decode()

    # read alignment SAM as a dataframe
    align_df = pd.read_csv(
        StringIO(out), sep="\t", skiprows=[0, 1, 2], header=None,
        usecols=[0, 3, 5, 9], names=["name", "start", "CIGAR", "seq"]
    )

    name_df = align_df["name"].str.split("_", expand=True)
    name_df = name_df[name_df.columns[-3:]]
    name_df.columns = ["amplicon_number", "alt_num_left", "alt_num_right"]
    return align_df.merge(name_df, left_index=True, right_index=True)
