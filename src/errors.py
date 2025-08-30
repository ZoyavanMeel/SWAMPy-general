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

ISSUE_6_BUG_CODE = -1

HP_FACTOR = 10000


def build_error_df(
    genome_abundances: dict, PATHS: dict, DEL_LENGTH_GEOMETRIC_PARAMETER: float,
    INS_MAX_LENGTH: int, VAF_DICT: dict, R_VAF_DICT: dict, DISALLOWED_POSITIONS: set[int],
    REF, U_SUBS_COUNT, U_INS_COUNT, U_DEL_COUNT, R_SUBS_COUNT, R_INS_COUNT, R_DEL_COUNT,
    SUBS_COUNT, INS_COUNT, DEL_COUNT, rng: np.random.Generator
):
    errors = pd.DataFrame(dict(errortype=["SUBS"]*SUBS_COUNT + ["DEL"]*DEL_COUNT + ["INS"]*INS_COUNT))

    # Z: recurrent = True; unique = False
    errors["recurrent"] = [True]*R_SUBS_COUNT + [False]*U_SUBS_COUNT + [True] * \
        R_DEL_COUNT + [False]*U_DEL_COUNT + [True]*R_INS_COUNT + [False]*U_INS_COUNT

    errors["mut_indices"] = errors.index
    errors["length"] = [1]*SUBS_COUNT + list(rng.geometric(p=DEL_LENGTH_GEOMETRIC_PARAMETER, size=DEL_COUNT)) + \
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


def amplicon_lookup(primer_df: pd.DataFrame, position: int, recurrent: bool) -> list[str]:
    # Find amplicons corresponding to a given position and sample in which the error (position) goes in case it's unique.

    mask = (position >= primer_df["Start_left"]) & (position <= primer_df["End_right"])
    corr_amps_df = primer_df.loc[mask, ["amplicon_number", "alt_num_left", "alt_num_right"]]
    # needs to be hashable for random.sample
    corresponding_amplicons = list(corr_amps_df.itertuples(index=False, name=None))

    if len(corresponding_amplicons) > 0 and not recurrent:
        return random.sample(corresponding_amplicons, k=1)
    return corresponding_amplicons


def add_high_frequency_errors(
    df_amplicons: pd.DataFrame, genome_abundances: dict,
    PATHS: dict[str, str], REF_NAME: str, RATES: dict[str, float],
    DEL_LENGTH_GEOMETRIC_PARAMETER: float, INS_MAX_LENGTH: int,
    VAF_DICT: dict[str, float], R_VAF_DICT: dict[str, float],
    DISALLOWED_POSITIONS: set[int], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    df_amplicons["var_num"] = 0
    df_amplicons["SNVs_in_primers"] = 0

    if SUBS_COUNT+INS_COUNT+DEL_COUNT == 0:
        return df_amplicons, "ERROR", "ERROR"

    # create a dataframe of errors that we want to introduce
    errors = build_error_df(
        genome_abundances, PATHS, DEL_LENGTH_GEOMETRIC_PARAMETER, INS_MAX_LENGTH,
        VAF_DICT, R_VAF_DICT, DISALLOWED_POSITIONS, REF, U_SUBS_COUNT, U_INS_COUNT,
        U_DEL_COUNT, R_SUBS_COUNT, R_INS_COUNT, R_DEL_COUNT, SUBS_COUNT, INS_COUNT, DEL_COUNT, rng
    )

    # all amplicons to be mutated
    error_amplicons = [amp for amp_list in errors["amplicons"] for amp in amp_list]

    # corresponding error index for that amplicon                 # str even though they're lists; that's just pandas
    indices = np.repeat(a=errors.index, repeats=errors["amplicons"].str.len())

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

    def amplicon_alignment(row: pd.Series) -> pd.Series:
        """Simple getter for the align_dfs so that I don't have to change too much of the original structure"""
        df = align_dfs[row["ref"]]
        # Z: If you're looling for time-saves, this is where! Because we need to index on the alt_nums as well as the
        #    amplicon_number (MultiIndex), this takes MUCH longer than only using the amplicon_number (if you ignore alts).
        #    There's most likely a way to do this faster?
        return df.loc[tuple(row[["amplicon_number", "alt_num_left", "alt_num_right"]].astype(str))]

    split_rows = []
    pulled_hyperparams_for_each_error = {}
    for _, row in df_amplicons.iterrows():
        # Sometimes more than 1 error are introduced to the same amplicon.
        # Find all errors that will be introduced to that amplicon
        mut_indices = [  # amp = (amp_num, alt_num_l, alt_num_r)
            indices[idx] for idx, amp in enumerate(error_amplicons)
            if amp[0] == row["amplicon_number"] and
            amp[1] == row["alt_num_left"] and
            amp[2] == row["alt_num_right"] and
            row["ref"] in errors.loc[indices[idx], "genome"]
        ]

        if len(mut_indices) == 0:
            split_rows.append(row)
            continue

        alignment = amplicon_alignment(row)

        # if the amplicon contains too many Ns it will not align, skip introducing PCR error to those
        # Z: with Bowtie2: won't align
        #    with BWA: soft-clip like crazy
        short_cigar = alignment["CIGAR"]
        if short_cigar == "*" or "S" in short_cigar:
            split_rows.append(row)
            continue

        # Start position and the sequence of the alignment(amplicon)
        start_p = alignment["start"]-1
        seq = alignment["seq"]

        # Record the CIGAR as a long string. i.e. "MMMII" instead of "3M2I"
        CIGAR = get_long_cigar(short_cigar)

        # Create an empty dataframe to hold how many reads each error combination will produce at the end.
        hyperparameters_df = pd.DataFrame()

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
                        f"""A PCR error is skipped since the position does not exist in the amplicon ({row['amplicon_filepath']}).
This is not a significant problem if you see only one of this warning. It likely means the amplicon shifted wrt to the reference and that caused the artificial error to not exist on the amplicon anymore."""
                    )
                    seq_pos.append(ISSUE_6_BUG_CODE)  # dummy placeholder (issue #6)

            # How many reads this specific error will have
            # Z: I changed this to take a percentage of the amplicon_hyperparameter instead of n_reads
            #    as it does in SWAMPy. This makes it so that the comments here talk about "reads" when
            #    that's not the case anymore.
            err_pos = errors.loc[mut_idx, "pos"]
            try:
                mut_hyperparameter = pulled_hyperparams_for_each_error[err_pos]
            except KeyError:
                mut_hyperparameter = rng.binomial(int(row["hyperparameter"]*HP_FACTOR), errors.loc[mut_idx, "VAF"])
                pulled_hyperparams_for_each_error[err_pos] = mut_hyperparameter
            # if number of reads and/or VAF are small, this can be 0
            if mut_hyperparameter != 0:
                # Take that many samples from the total of the imaginary reads of that amplicon
                hyperparameter_prop = sorted(
                    random.sample(range(int(row["hyperparameter"]*HP_FACTOR)), k=mut_hyperparameter)
                )
                hyperparameter_prop = [str(a) for a in hyperparameter_prop]
                hyperparameter_prop = [a+"," for a in hyperparameter_prop]  # "," will be used for grouping

                muts = [str(mut_idx)]*mut_hyperparameter  # keep track of the mutation
                muts = [a+"," for a in muts]

                hyperparameter_df = pd.DataFrame(dict(hyperparameter=hyperparameter_prop, muts=muts))
                hyperparameters_df = pd.concat([hyperparameters_df, hyperparameter_df], ignore_index=True)

        if not hyperparameters_df.empty:
            # group wrt imaginary reads to see which ones ended up with wich errors
            rows_split_per_variation = split_hyperparameter_per_error_variation(
                row, hyperparameters_df, seq, seq_pos, mut_indices,
                errors, PATHS
            )
            split_rows.extend(rows_split_per_variation)
        else:
            # if there are no errors with a non-zero count, skip introducing errors to that amplicon
            split_rows.append(row)

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
    expanded_df_amplicons = pd.DataFrame(split_rows)
    return expanded_df_amplicons, vcf_errordf


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


def get_mutations_in_primers(amplicon: pd.Series, mutation_df: pd.DataFrame) -> int:
    """
    Return how many SNVs are in the primer regions of the given amplicon.
    Coordinates are based on the amplicon (`5'->3': 0->len(amplicon)-1`)
    """
    amp_length = amplicon["right"] + amplicon["right_primer_length"] - amplicon["left"]
    return ((
        mutation_df["seq_pos"] <= amplicon["left_primer_length"]-1
    ) | (
        mutation_df["seq_pos"] >= amp_length-1 - amplicon["right_primer_length"]
    )).sum().sum()


def get_sequence_params(row: pd.Series, REFSEQ, VAF_DICT: dict, R_VAF_DICT: dict, genome_abundances: dict, rng: np.random.Generator) -> pd.Series:
    ref = REFSEQ.seq[row["pos"]] if row["errortype"] != "DEL" else REFSEQ.seq[row["pos"]-1:row["pos"]+row["length"]]
    alt = alts(ref, row["errortype"], row["length"])

    if not row["recurrent"]:
        vaf = rng.dirichlet(VAF_DICT[row["errortype"]], size=None)[0]
        genome = random.choices(
            population=list(genome_abundances.keys()),
            weights=list(genome_abundances.values()),
            k=1
        )
    else:
        vaf = rng.dirichlet(R_VAF_DICT[row["errortype"]], size=None)[0]
        genome = list(genome_abundances.keys())
    return pd.Series([genome, ref, alt, vaf])


def split_hyperparameter_per_error_variation(amp_row: pd.Series, hyperparameters_df: pd.DataFrame, seq: str, seq_pos: list, mut_indices: list, errors: pd.DataFrame, paths: dict[str, str]) -> list[pd.Series]:
    """This has all the hyperparameters multiplied by HP_FACTOR so the original code still works"""
    hyperparameters_df = hyperparameters_df.groupby("hyperparameter").sum()
    hyperparameters_df["count"] = 1

    # group by different combinations of errors to count how many reads each combination will produce.
    hyperparameters_df = hyperparameters_df.groupby("muts", as_index=False).sum()
    # remove the , and turn tham into a list
    hyperparameters_df["muts"] = hyperparameters_df.apply(lambda x: x["muts"].split(",")[:-1], axis=1)

    # create a dataframe of all errors of the amplicon. Contains pos, mut_index, errortype, length, alt
    seq_pos_df = pd.DataFrame(dict(seq_pos=seq_pos, mut_indices=mut_indices))
    seq_pos_df = seq_pos_df.merge(errors[["errortype", "mut_indices", "length", "alt"]], on="mut_indices")
    # skip errors that correspond to deletions (issue #6)
    seq_pos_df = seq_pos_df[seq_pos_df['seq_pos'] != ISSUE_6_BUG_CODE]
    # amplicon's number of reads - total count of all error combination versions is the count of non-mutated (old) version.
    amp_row["hyperparameter"] -= sum(hyperparameters_df["count"])/HP_FACTOR
    split_rows = [amp_row]

    # for all error combination versions of the amplicon
    # Z: `hyperparameters_df` contains the mutations that will be added to this variation of this amplicon
    #    There are multiple variants of each amplicon. Some have multiple errors, some only have
    #    one, some only the other. This is what `_p{idx+1}` indicates: a unique version of that
    #    amplicon. If there is no `_p` suffix, then that's the original with no errors.
    for idx, pcr_error in enumerate(hyperparameters_df.itertuples()):
        # Z: watch out: linters might not understand how itertuples works (like mine)
        # if a specific combination has 0 reads, pass
        if pcr_error.count == 0:
            continue

        # create a final mutation_df that contains all the errors in that specific combination
        mutation_df = seq_pos_df.loc[seq_pos_df["mut_indices"].isin([int(a) for a in pcr_error.muts])]
        mutation_df = mutation_df.sort_values("seq_pos")
        mutation_df.reset_index(drop=True, inplace=True)

        # introduce those errors one by one.
        new_seq = mutate_amplicon(seq, mutation_df)

        # add the new amplicon to the list
        new_path = amp_row["amplicon_filepath"][:-6] + "_p" + str(idx+1) + ".fasta"

        new_row = amp_row.copy()
        new_row["amplicon_filepath"] = new_path
        new_row["var_num"] = idx+1
        new_row["SNVs_in_primers"] = get_mutations_in_primers(new_row, mutation_df)
        new_row["hyperparameter"] = pcr_error.count / HP_FACTOR  # remember the multiplication, young padawan
        split_rows.append(new_row)

        # write the fasta file of the new amplicon.
        # Name all the PCR error combinations as _p1, _p2 and etc.
        with open(os.path.join(paths['AMPLICONS_FOLDER'], new_path), "w") as new_amp:
            if len(new_seq) < 250:
                print(new_row)
                print(new_seq)
            new_amp.write(f">{new_path[:-6]}\n")
            new_amp.write(new_seq + "\n")
    return split_rows


def mutate_amplicon(seq: str, mutation_df: pd.DataFrame) -> str:
    """Applies all errors/mutations in the dataframe to the given sequence (amplicon)"""
    new_seq = ""
    for indx, mutation in enumerate(mutation_df.itertuples()):
        # the part up to the first error is the same
        if indx == 0:
            new_seq += seq[0:mutation.seq_pos]

            # if an error is substition or indel, take the part up to and excluding the error position
            # add alternative instead of the ref at error pos.
        if mutation.errortype == "SUBS" or mutation.errortype == "INS":
            new_seq = new_seq+mutation.alt
            # then add the part up to the next error
            if indx + 1 in mutation_df.index:
                new_seq += seq[mutation.seq_pos+1:mutation_df.loc[indx+1, "seq_pos"]]
                # if it is the last error, add all the remaining sequence
            else:
                new_seq += seq[mutation.seq_pos+1:]

        else:  # DEL
            # if it is a deletion add the next section of the sequence but leave out the first n bases of it
            if indx+1 in mutation_df.index:
                end_pos = mutation_df.loc[indx+1, "seq_pos"]
                subseq = seq[mutation.seq_pos+1:end_pos]
            else:
                subseq = seq[mutation.seq_pos+1:]

            new_seq += subseq[mutation.length-1:]

    # This can happen if mutation_df is empty, which can happen because of the ISSUE_6_BUG_CODE
    if new_seq == "":
        return seq
    return new_seq


def get_long_cigar(short_cigar: str):
    long_cigar = ""
    for idx, cigar in enumerate(re.split("(M|D|I)", short_cigar)[:-1]):
        if idx % 2 == 0:
            prev = cigar[:]
        else:
            long_cigar += cigar*int(prev)
    return long_cigar


def align_amps_to_ref(PATHS: dict[str, str], amp_path: str):
    amp_path = re.sub(r"[/,\|&]", r"&", amp_path).replace(" ", "_")
    alignment = subprocess.run(
        ["bwa", "mem", "-L", "1000000", PATHS['INDEX_BASE'], f"{PATHS['AMPLICONS_FOLDER']}/{amp_path}"],
        # ["bowtie2", "-x", PATHS['INDEX_BASE'], "-f", f"{PATHS['AMPLICONS_FOLDER']}/{amp_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = alignment.stdout.decode(), alignment.stderr.decode()
    hp.check_stderr(err, "BWA")

    # filter out supplementary alignments
    samview = subprocess.run(
        ["samtools", "view", "-F", "0x800", "-"],
        input=alignment.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = samview.stdout.decode()

    # read alignment SAM as a dataframe
    align_df = pd.read_csv(
        StringIO(out), sep="\t", header=None,
        usecols=[0, 3, 5, 9], names=["name", "start", "CIGAR", "seq"]
    )

    name_df = align_df["name"].str.split("_", expand=True)
    name_df = name_df[name_df.columns[-3:]]
    name_df.columns = ["amplicon_number", "alt_num_left", "alt_num_right"]
    return align_df.merge(
        name_df, left_index=True, right_index=True
    ).set_index(
        ["amplicon_number", "alt_num_left", "alt_num_right"]
    )
