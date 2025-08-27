import argparse
from os.path import dirname, join, abspath, basename
from time import perf_counter
import os
import logging
from Bio import SeqIO
import pandas as pd
import numpy as np
import random
import shutil

from art_runner import art_illumina
from create_amplicons import get_alignment_df_and_call_SNVs, write_amplicon
from biases import load_amp_dist_file, apply_bias, correct_dropout_rate
from errors import add_high_frequency_errors
import helpers as hp

# All these caps variables are set once (by user inputs, with default values) but then never touched again.
BASE_DIR = join(dirname(dirname(abspath(__file__))), "example")
TEMP_FOLDER = join(BASE_DIR, "temp")
GENOMES_FILE = join(BASE_DIR, "genomes.fasta")
ABUNDANCES_FILE = join(BASE_DIR, "abundances.tsv")
PRIMER_SET = "a1"
PRIMER_SET_FOLDER = join(dirname(dirname(abspath(__file__))), "primer_sets")
OUTPUT_FOLDER = os.getcwd()
OUTPUT_FILENAME_PREFIX = "example"
N_READS = 100000
READ_LENGTH = 250
SEQ_SYS = "MSv3"
QPROF1 = None
QPROF2 = None
SEED = np.random.SeedSequence()
RNG = np.random.default_rng(SEED)
DISALLOWED_POSITIONS = {}
FRAGMENT_AMPLICONS = False
FRAGMENT_LEN_MEAN = 0
FRAGMENT_LEN_SD = 0
ART_QSHIFT = 0

# PCR-error related variables:
REFERENCE = join(dirname(dirname(abspath(__file__))), "ref", "MN908947.3.fasta")
REF_NAME = "MN908947.3"
REF_LEN = 29903
INDEX_BASE = REFERENCE.strip(".fasta").strip("fa")

# wastewater settings
MUT_RATE_SCALING = 5

U_SUBS_RATE = 0.002485 * MUT_RATE_SCALING
U_INS_RATE = 0.00002 * MUT_RATE_SCALING
U_DEL_RATE = 0.000115 * MUT_RATE_SCALING
R_SUBS_RATE = 0.003357 * MUT_RATE_SCALING
R_INS_RATE = 0.00002 * MUT_RATE_SCALING
R_DEL_RATE = 0 * MUT_RATE_SCALING

DEL_LENGTH_GEOMETRIC_PARAMETER = 0.69
INS_MAX_LENGTH = 14

# clinical   = 0.100
# wastewater = 0.133
DROPOUT_RATE = 0.100

SUBS_VAF_DIRICHLET_PARAMETER = "0.29,1.89"
INS_VAF_DIRICHLET_PARAMETER = "0.33,0.45"
DEL_VAF_DIRICHLET_PARAMETER = "0.59,0.41"

R_SUBS_VAF_DIRICHLET_PARAMETER = SUBS_VAF_DIRICHLET_PARAMETER
R_INS_VAF_DIRICHLET_PARAMETER = INS_VAF_DIRICHLET_PARAMETER
R_DEL_VAF_DIRICHLET_PARAMETER = DEL_VAF_DIRICHLET_PARAMETER

SNV_BALANCE = 0.5
SNV_DIRICHLET_PARAMETER = 200
AMPLICON_DIRICHLET_PARAMETER = 200


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SARS-CoV-2 metagenome simulation.")
    parser.add_argument("--genomes_file", "-g",
                        help="File containing all of the genome lineages to simulate", default=GENOMES_FILE)
    parser.add_argument("--reference", "-r",
                        help="File containing the reference sequence. Default: ../ref/MN908947.3.fasta", default=REFERENCE)
    parser.add_argument("--index_base",
                        help="BWA reference index folder (bwa_index_base). Default: ../ref/MN908947.3", default=INDEX_BASE)
    parser.add_argument("--temp_folder", "-t",
                        help="A path for a temporary output folder to store intemediate files. Including FASTA files of genomes, amplicons, and their BWA indices", default=TEMP_FOLDER)
    parser.add_argument("--genome_abundances", "-a",
                        help="TSV of genome abundances.", default=ABUNDANCES_FILE)
    parser.add_argument("--primer_set", help="Primer set. This sets defaults for the parameters, --primers_file, --primer_bed, and --amplicon_distribution_file, which are overwritten if separately provided. Can be either a1 for Artic v1, a4 for Artic v4, a5 for Artic v5.3, and n2 for Nimagen v2, or c for custom (custom provides no defaults, so each of --primers_file, --primer_bed, and --amplicon_distribution_file must be provided separately)",
                        required=True, choices=["a1", "a4", "a5", "n2", "c"])
    parser.add_argument("--snv_balance", "-b", help="Indicates the balance between the given amplicon distribution and the calculated SNV-bias. Default: 0.5 (50/50). Max/min = 1.0/0.0. Increasing this parameter increases the weight of the SNV-bias",
                        default=SNV_BALANCE)
    parser.add_argument("--primers_file",
                        help="Fastq file with formatted names of primers - see primer_sets folder for examples. Only needed if using --primer_set=custom.", default=None)
    parser.add_argument("--primer_bed",
                        help="bed formatted file of primers to use, see primer_sets folder for examples. Only needed if using --primer_bed=custom", default=None)
    parser.add_argument("--amplicon_distribution_file",
                        help="TSV file of a prior for amplicon proportions, see primer_sets folder for examples. Set only if using --primer_bed=custom. When unset: assume equal distribution between amplicons.", default=None)
    parser.add_argument("--output_folder", "-o",
                        help="A path for a folder where the output fastq files will be stored. Default is working directory", default=OUTPUT_FOLDER)
    parser.add_argument("--output_filename_prefix", "-x",
                        help="Name of the fastq files name1.fastq, name2.fastq", default=OUTPUT_FILENAME_PREFIX)
    parser.add_argument("--seqSys", help="Name of the sequencing system, options to use are given by the art_illumina help text, and are:" +
                        """GA1 - GenomeAnalyzer I (36bp,44bp), GA2 - GenomeAnalyzer II (50bp, 75bp)
           HS10 - HiSeq 1000 (100bp),          HS20 - HiSeq 2000 (100bp),      HS25 - HiSeq 2500 (125bp, 150bp)
           HSXn - HiSeqX PCR free (150bp),     HSXt - HiSeqX TruSeq (150bp),   MinS - MiniSeq TruSeq (50bp)
           MSv1 - MiSeq v1 (250bp),            MSv3 - MiSeq v3 (250bp),        NS50 - NextSeq500 v2 (75bp), or custom - in which case
           you need to pass in two custom (ART) quality score profiles using --qprof1 and --qprof2""", default=SEQ_SYS)
    parser.add_argument(
        "--qprof1", help="Custom quality score profile for R1 reads (ART) - use with --seqSys=custom", default=QPROF1)
    parser.add_argument(
        "--qprof2", help="Custom quality score profile for R1 reads (ART) - use with --seqSys=custom", default=QPROF2)
    parser.add_argument(
        "--n_reads", "-n", help="Approximate number of reads in fastq file (subject to sampling stochasticity).", default=N_READS)
    parser.add_argument("--dropout_rate", "-d",
                        help="Approximate percentage of amplicons dropped (subject to sampling stochasticity).", default=DROPOUT_RATE)
    parser.add_argument("--read_length", "-l",
                        help="Length of reads taken from the sequencing machine.", default=READ_LENGTH)
    parser.add_argument("--seed", "-s", help="Random seed integer (must be non-negative)")
    parser.add_argument("--quiet", "-q", help="Add this flag to supress verbose output.", action='store_true')
    parser.add_argument("--fragment_amplicons", help="Cut amplicons randomly into fragments when running ART for sequencing errors (set as True or False, default is False).",
                        action='store_true', default=FRAGMENT_AMPLICONS)
    parser.add_argument("--fragment_len_mean",
                        help="Mean fragment length if using --fragment_amplicons", default=FRAGMENT_LEN_MEAN)
    parser.add_argument(
        "--fragment_len_sd", help="Standard deviation of fragment lengths if using --fragment_amplicons", default=FRAGMENT_LEN_SD)
    parser.add_argument("--amplicon_dirichlet_parameter", "-c", default=AMPLICON_DIRICHLET_PARAMETER)
    parser.add_argument("--snv_dirichlet_parameter", "-v", default=SNV_DIRICHLET_PARAMETER)
    parser.add_argument("--autoremove", action='store_true', help="Delete temproray files after execution.")
    parser.add_argument("--no_pcr_errors", action='store_true',
                        help="Turn off PCR errors. The output will contain only sequencing errors. Other PCR-error related options will be ignored")
    parser.add_argument(
        "--art_qshift", help="Supply ART with --qShift and --qShift2 parameters (bumps up quality scores).", default=ART_QSHIFT)
    parser.add_argument("--unique_insertion_rate", "-ins",
                        help="PCR insertion error rate. Unique to one source genome in the mixture Default is 0.00002", default=U_INS_RATE)
    parser.add_argument("--unique_deletion_rate", "-del",
                        help="PCR deletion error rate. Unique to one source genome in the mixture Default is 0.000115", default=U_DEL_RATE)
    parser.add_argument("--unique_substitution_rate", "-subs",
                        help="PCR substitution error rate. Unique to one source genome in the mixture Default is 0.002485", default=U_SUBS_RATE)
    parser.add_argument("--recurrent_insertion_rate", "-rins",
                        help="PCR insertion error rate. Recurs across source genomes. Default is 0.00002", default=R_INS_RATE)
    parser.add_argument("--recurrent_deletion_rate", "-rdel",
                        help="PCR deletion error rate. Recurs across source genomes. Default is 0", default=R_DEL_RATE)
    parser.add_argument("--recurrent_substitution_rate", "-rsubs",
                        help="PCR substitution error rate. Recurs across source genomes. Default is 0.003357", default=R_SUBS_RATE)
    parser.add_argument("--deletion_length_p", "-dl",
                        help="Geometric distribution parameter, p, for PCR deletion length. Default is 0.69", default=DEL_LENGTH_GEOMETRIC_PARAMETER)
    parser.add_argument("--max_insertion_length", "-il",
                        help="Maximum PCR insertion length in bases (uniform distribution boundry). Default is 14", default=INS_MAX_LENGTH)
    parser.add_argument("--subs_VAF_alpha", "-sv",
                        help="alpha1,alpha2 of the Dirichlet distribution for VAF of the unique PCR error. Default is 0.29,1.89", default=SUBS_VAF_DIRICHLET_PARAMETER)
    parser.add_argument("--del_VAF_alpha", "-dv",
                        help="alpha1,alpha2 of the Dirichlet distribution for VAF of the unique PCR error. Default is 0.59,0.41", default=DEL_VAF_DIRICHLET_PARAMETER)
    parser.add_argument("--ins_VAF_alpha", "-iv",
                        help="alpha1,alpha2 of the Dirichlet distribution for VAF of the unique PCR error. Default is 0.33,0.45", default=INS_VAF_DIRICHLET_PARAMETER)
    parser.add_argument("--r_subs_VAF_alpha", "-rsv",
                        help="alpha1,alpha2 of the Dirichlet distribution for VAF of the recurrent PCR error. Default is equal to unique erros", default=SUBS_VAF_DIRICHLET_PARAMETER)
    parser.add_argument("--r_del_VAF_alpha", "-rdv",
                        help="alpha1,alpha2 of the Dirichlet distribution for VAF of the recurrent PCR error. Default is equal to unique erros", default=DEL_VAF_DIRICHLET_PARAMETER)
    parser.add_argument("--r_ins_VAF_alpha", "-riv",
                        help="alpha1,alpha2 of the Dirichlet distribution for VAF of the recurrent PCR error. Default is equal to unique erros", default=INS_VAF_DIRICHLET_PARAMETER)
    parser.add_argument("--disallowed_positions", "-dis",
                        help="A comma separated list of 0 based genome coordinates (relative to the reference genome) where substitutions and deletions are not allowed.", default=DISALLOWED_POSITIONS)
    return parser


def load_command_line_args() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    global TEMP_FOLDER
    TEMP_FOLDER = args.temp_folder
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER, exist_ok=True)

    global GENOMES_FOLDER
    GENOMES_FOLDER = join(TEMP_FOLDER, "genomes")
    if not os.path.exists(GENOMES_FOLDER):
        os.makedirs(GENOMES_FOLDER, exist_ok=True)

    global REFERENCE
    REFERENCE = args.reference
    global INDEX_BASE
    INDEX_BASE = args.index_base
    # Z: not the most elegant, but whatevs
    ref = SeqIO.read(REFERENCE, format="fasta")
    global REF_LEN
    REF_LEN = len(ref)
    global REF_NAME
    REF_NAME = ref.name
    del ref

    global SNV_BALANCE
    SNV_BALANCE = float(args.snv_balance)
    if SNV_BALANCE > 1.0 or SNV_BALANCE < 0.0:
        raise ValueError(f"snv_balance parameter can only be between 0.0 and 1.0, but was {SNV_BALANCE}")

    global DROPOUT_RATE
    DROPOUT_RATE = float(args.dropout_rate)
    if DROPOUT_RATE > 1.0 or DROPOUT_RATE < 0.0:
        raise ValueError(f"dropout_rate parameter can only be between 0.0 and 1.0, but was {SNV_BALANCE}")

    global GENOMES_FILE
    GENOMES_FILE = args.genomes_file
    global GENOMES_FILE2
    GENOMES_FILE2 = join(GENOMES_FOLDER, basename(GENOMES_FILE))

    global AMPLICONS_FOLDER
    AMPLICONS_FOLDER = join(TEMP_FOLDER, "amplicons")
    if not os.path.exists(AMPLICONS_FOLDER):
        os.makedirs(AMPLICONS_FOLDER, exist_ok=True)

    global INDICES_FOLDER
    INDICES_FOLDER = join(TEMP_FOLDER, "indices")
    if not os.path.exists(INDICES_FOLDER):
        os.mkdir(INDICES_FOLDER)

    global ABUNDANCES_FILE
    ABUNDANCES_FILE = args.genome_abundances
    global ABUNDANCES_FILE2
    ABUNDANCES_FILE2 = join(GENOMES_FOLDER, basename(ABUNDANCES_FILE))

    global OUTPUT_FOLDER
    OUTPUT_FOLDER = args.output_folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    global OUTPUT_FILENAME_PREFIX
    OUTPUT_FILENAME_PREFIX = args.output_filename_prefix
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME_PREFIX}.log")),
            logging.StreamHandler()
        ]
    )

    global PRIMER_SET
    PRIMER_SET = args.primer_set

    global PRIMERS_FILE
    global AMPLICON_DISTRIBUTION_FILE
    global PRIMER_BED

    if PRIMER_SET == "a1":
        PRIMERS_FILE = join(PRIMER_SET_FOLDER, "artic_v3_primers_no_alts.fastq")
        logging.info(f"Primer set: Artic v1")
        PRIMER_BED = join(PRIMER_SET_FOLDER, "articV3_no_alt.bed")
        AMPLICON_DISTRIBUTION_FILE = join(PRIMER_SET_FOLDER, "artic_v3_amplicon_distribution.tsv")

    elif PRIMER_SET == "a4":
        PRIMERS_FILE = join(PRIMER_SET_FOLDER, "artic_v4_primers.fastq")
        logging.info(f"Primer set: Artic v4")
        PRIMER_BED = join(PRIMER_SET_FOLDER, "articV4.bed")
        AMPLICON_DISTRIBUTION_FILE = join(PRIMER_SET_FOLDER, "artic_v4_amplicon_distribution.tsv")

    elif PRIMER_SET == "a5":
        PRIMERS_FILE = join(PRIMER_SET_FOLDER, "artic_v5.3_primers.fastq")
        logging.info(f"Primer set: Artic v5.3")
        PRIMER_BED = join(PRIMER_SET_FOLDER, "articV5.3.bed")
        AMPLICON_DISTRIBUTION_FILE = join(PRIMER_SET_FOLDER, "artic_v5.3_amplicon_distribution.tsv")

    elif PRIMER_SET == "n2":
        PRIMERS_FILE = join(PRIMER_SET_FOLDER, "nimagen_v2_primers.fastq")
        logging.info(f"Primer set: Nimagen v2")
        PRIMER_BED = join(PRIMER_SET_FOLDER, "nimagenV2.bed")
        AMPLICON_DISTRIBUTION_FILE = join(PRIMER_SET_FOLDER, "nimagen_v2_amplicon_distribution.tsv")

    elif PRIMER_SET == "c":
        PRIMERS_FILE = args.primers_file
        logging.info("Primer set: Custom")
        PRIMER_BED = args.primer_bed
        AMPLICON_DISTRIBUTION_FILE = args.amplicon_distribution_file

    if args.primers_file:
        PRIMERS_FILE = args.primers_file
    if args.primer_bed:
        PRIMER_BED = args.primer_bed
    if args.amplicon_distribution_file:
        AMPLICON_DISTRIBUTION_FILE = args.amplicon_distribution_file

    global N_READS
    N_READS = int(args.n_reads)
    logging.info(f"Number of reads: {N_READS}")

    global READ_LENGTH
    READ_LENGTH = int(args.read_length)

    global SEED
    global RNG
    if args.seed is not None:
        SEED = np.random.SeedSequence(np.abs(int(args.seed)))
        RNG = np.random.default_rng(SEED)

    random.seed(int(SEED.entropy))
    logging.info(f"Random seed: {int(SEED.entropy)}")

    global VERBOSE
    VERBOSE = not args.quiet

    global FRAGMENT_AMPLICONS
    FRAGMENT_AMPLICONS = args.fragment_amplicons

    global FRAGMENT_LEN_MEAN
    global FRAGMENT_LEN_SD

    global SEQ_SYS
    SEQ_SYS = args.seqSys
    global QPROF1
    QPROF1 = args.qprof1
    global QPROF2
    QPROF2 = args.qprof2

    if SEQ_SYS.lower() == "custom":
        if (not QPROF1) or (not QPROF2):
            logging.error("If you supply --seqSys custom then you must supply --qprof1 and --qprof2 files. Exiting.")
            exit("If you supply --seqSys custom then you must supply --qprof1 and --qprof2 files. Exiting.")

    FRAGMENT_LEN_MEAN = float(args.fragment_len_mean)
    FRAGMENT_LEN_SD = float(args.fragment_len_sd)

    if FRAGMENT_AMPLICONS:
        if FRAGMENT_LEN_MEAN < READ_LENGTH:
            logging.error(
                "If you plan to fragment your amplicons (--fragment_amplicons=True), you must set --fragment_len_mean and --fragment_len_sd")
            logging.error("The mean fragment length must be greater than the read length.")
            logging.error(
                f"Currently the mean fragment length is {FRAGMENT_LEN_MEAN}, the read length is {READ_LENGTH}.")
            exit(1)

    global AMPLICON_DIRICHLET_PARAMETER
    AMPLICON_DIRICHLET_PARAMETER = int(args.amplicon_dirichlet_parameter)
    logging.info(f"Amplicon dirichlet_parameter: {AMPLICON_DIRICHLET_PARAMETER}")

    global SNV_DIRICHLET_PARAMETER
    SNV_DIRICHLET_PARAMETER = int(args.snv_dirichlet_parameter)
    logging.info(f"SNV dirichlet_parameter: {SNV_DIRICHLET_PARAMETER}")

    global AUTOREMOVE
    AUTOREMOVE = args.autoremove

    # PCR error arguments
    global NO_PCR_ERRORS
    NO_PCR_ERRORS = args.no_pcr_errors

    global ART_QSHIFT
    ART_QSHIFT = int(args.art_qshift)

    global U_SUBS_RATE
    U_SUBS_RATE = float(args.unique_substitution_rate)

    global U_INS_RATE
    U_INS_RATE = float(args.unique_insertion_rate)

    global U_DEL_RATE
    U_DEL_RATE = float(args.unique_deletion_rate)

    global R_SUBS_RATE
    R_SUBS_RATE = float(args.recurrent_substitution_rate)

    global R_INS_RATE
    R_INS_RATE = float(args.recurrent_insertion_rate)

    global R_DEL_RATE
    R_DEL_RATE = float(args.recurrent_deletion_rate)

    global DISALLOWED_POSITIONS
    if len(args.disallowed_positions) != 0:
        DISALLOWED_POSITIONS = {int(x) for x in args.disallowed_positions.split(",")}

    global DEL_LENGTH_GEOMETRIC_PARAMETER
    DEL_LENGTH_GEOMETRIC_PARAMETER = float(args.deletion_length_p)

    global INS_MAX_LENGTH
    INS_MAX_LENGTH = int(args.max_insertion_length)

    global SUBS_VAF_DIRICHLET_PARAMETER
    SUBS_VAF_DIRICHLET_PARAMETER = args.subs_VAF_alpha.split(",")
    if len(SUBS_VAF_DIRICHLET_PARAMETER) != 2:
        logging.error(
            f"subs_VAF_alpha argument must be a list of 2 values seperated by comma. Example: 0.5,0.4. You entered {args.subs_VAF_alpha} ")
        exit(1)
    else:
        SUBS_VAF_DIRICHLET_PARAMETER = [float(a) for a in SUBS_VAF_DIRICHLET_PARAMETER]

    global INS_VAF_DIRICHLET_PARAMETER
    INS_VAF_DIRICHLET_PARAMETER = args.ins_VAF_alpha.split(",")
    if len(INS_VAF_DIRICHLET_PARAMETER) != 2:
        logging.error(
            f"ins_VAF_alpha argument must be a list of 2 values seperated by comma. Example: 0.5,0.4. You entered {args.ins_VAF_alpha} ")
        exit(1)
    else:
        INS_VAF_DIRICHLET_PARAMETER = [float(a) for a in INS_VAF_DIRICHLET_PARAMETER]

    global DEL_VAF_DIRICHLET_PARAMETER
    DEL_VAF_DIRICHLET_PARAMETER = args.del_VAF_alpha.split(",")
    if len(DEL_VAF_DIRICHLET_PARAMETER) != 2:
        logging.error(
            f"del_VAF_alpha argument must be a list of 2 values seperated by comma. Example: 0.5,0.4. You entered {args.del_VAF_alpha} ")
        exit(1)
    else:
        DEL_VAF_DIRICHLET_PARAMETER = [float(a) for a in DEL_VAF_DIRICHLET_PARAMETER]

    global R_SUBS_VAF_DIRICHLET_PARAMETER
    R_SUBS_VAF_DIRICHLET_PARAMETER = args.r_subs_VAF_alpha.split(",")
    if len(R_SUBS_VAF_DIRICHLET_PARAMETER) != 2:
        logging.error(
            f"r_subs_VAF_alpha argument must be a list of 2 values seperated by comma. Example: 0.5,0.4. You entered {args.r_subs_VAF_alpha} ")
        exit(1)
    else:
        R_SUBS_VAF_DIRICHLET_PARAMETER = [float(a) for a in R_SUBS_VAF_DIRICHLET_PARAMETER]

    global R_INS_VAF_DIRICHLET_PARAMETER
    R_INS_VAF_DIRICHLET_PARAMETER = args.r_ins_VAF_alpha.split(",")
    if len(R_INS_VAF_DIRICHLET_PARAMETER) != 2:
        logging.error(
            f"r_ins_VAF_alpha argument must be a list of 2 values seperated by comma. Example: 0.5,0.4. You entered {args.r_ins_VAF_alpha} ")
        exit(1)
    else:
        R_INS_VAF_DIRICHLET_PARAMETER = [float(a) for a in R_INS_VAF_DIRICHLET_PARAMETER]

    global R_DEL_VAF_DIRICHLET_PARAMETER
    R_DEL_VAF_DIRICHLET_PARAMETER = args.r_del_VAF_alpha.split(",")
    if len(R_DEL_VAF_DIRICHLET_PARAMETER) != 2:
        logging.error(
            f"r_del_VAF_alpha argument must be a list of 2 values seperated by comma. Example: 0.5,0.4. You entered {args.r_del_VAF_alpha} ")
        exit(1)
    else:
        R_DEL_VAF_DIRICHLET_PARAMETER = [float(a) for a in R_DEL_VAF_DIRICHLET_PARAMETER]

    global VAF_PARAMETER_DICT
    VAF_PARAMETER_DICT = {
        "SUBS": SUBS_VAF_DIRICHLET_PARAMETER,
        "INS": INS_VAF_DIRICHLET_PARAMETER,
        "DEL": DEL_VAF_DIRICHLET_PARAMETER
    }
    global R_VAF_PARAMETER_DICT
    R_VAF_PARAMETER_DICT = {
        "SUBS": R_SUBS_VAF_DIRICHLET_PARAMETER,
        "INS": R_INS_VAF_DIRICHLET_PARAMETER,
        "DEL": R_DEL_VAF_DIRICHLET_PARAMETER
    }

    global RATES
    RATES = {
        "U_SUBS_RATE": U_SUBS_RATE,
        "U_INS_RATE": U_INS_RATE,
        "U_DEL_RATE": U_DEL_RATE,
        "R_SUBS_RATE": R_SUBS_RATE,
        "R_INS_RATE": R_INS_RATE,
        "R_DEL_RATE": R_DEL_RATE
    }


def remove_tmp(temp_folder: str):
    for filename in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def format_ART_input(lineages_formatted: list, amplicons: list[str], n_reads: list[int], AMPLICONS_FOLDER: str) -> tuple[list[str], list[int]]:
    """Merge art_illumina runs which have the same read count to optimize"""
    merged_n_reads = list(set(n_reads))
    if merged_n_reads[0] == 0:
        merged_n_reads = merged_n_reads[1:]

    merged_amplicons = [join(AMPLICONS_FOLDER, f"merged_amplicon_rcount_{a}.fasta") for a in merged_n_reads]

    # Z: First read the amplicon fasta as a whole to simplify jumping around.
    #    Order can't(?) be guarantueed due to dropout and such
    #    Can't be bothered to change much about the original structure either
    with open(join(AMPLICONS_FOLDER, "all_original_amplicons.fasta"), "w") as af:
        for lineage in lineages_formatted:
            with open(join(AMPLICONS_FOLDER, f"all_amplicons_{lineage}.fasta"), "r") as lf:
                shutil.copyfileobj(lf, af)
                # These files were all written by SWAMPy, so we know for sure they don't end on a newline
                af.write("\n")

    #    I don't wanna hear ANYTHING about this thing's efficiency
    with open(join(AMPLICONS_FOLDER, "all_original_amplicons.fasta"), "r") as af:
        line_offset = {}
        offset = 0
        for line in af:
            tmp = line.strip().split("_")
            if len(tmp) < 2:
                offset += len(line)
                continue
            amp_info = tmp[-3:]
            name = "_".join(tmp[:-3]).replace(" ", "&").replace("/", "&").replace(",", "&")
            line_offset.update({tuple([name]+amp_info): offset})
            offset += len(line)
        af.seek(0)

        for readcount, m_amplicon in zip(merged_n_reads, merged_amplicons):
            with open(m_amplicon, "w") as merged_handle:
                for amp in [amplicons[idx] for idx, r in enumerate(n_reads) if r == readcount]:
                    try:
                        with open(join(AMPLICONS_FOLDER, amp), "r") as amp_filehandle:
                            shutil.copyfileobj(amp_filehandle, merged_handle)
                    except FileNotFoundError as e:
                        # these are the non-'p' files
                        tmp = amp.strip(".fasta").split("_")
                        amp_info = tmp[-3:]
                        name = ">" + "_".join(tmp[:-3])
                        af.seek(line_offset[tuple([name]+amp_info)])
                        # each fasta is two lines: name, sequence
                        merged_handle.write(af.readline())
                        merged_handle.write(af.readline())
    return merged_amplicons, merged_n_reads


if __name__ == "__main__":
    t1 = perf_counter()

    # STEP 0: Read command line arguments
    load_command_line_args()

    # Change spaces with "_" in genomes fasta file and record as a different file.
    os.system(f"sed 's/ /_/g' {GENOMES_FILE} > {GENOMES_FILE2}")
    os.system(f"sed 's/ /_/g' {ABUNDANCES_FILE} > {ABUNDANCES_FILE2}")
    if VERBOSE:
        logging.info("Spaces in the genomes and abundances files are processed as '_' characters if exist")

    # STEP 1: Simulate Viral Population

    # Read genome abundances csv file
    genome_abundances = {}
    amplicon_df = pd.DataFrame()

    with open(ABUNDANCES_FILE2) as ab_file:
        for line in ab_file:
            name, relative_abundance = tuple(line.split("\t"))
            genome_abundances[name] = float(relative_abundance)

    total = sum(genome_abundances.values())
    if abs(total - 1) > 1e-9:
        if total <= 0:
            logging.info(f"The total genome abundance is set to {total}, which is impossible.")
            exit(1)

        logging.info(f"Total of relative abundance values is {total}, not 1.")
        logging.info("Continuing, normalising total of genome abundances to 1.")

        for k in genome_abundances.keys():
            genome_abundances[k] /= total

    n_genomes = len(genome_abundances)

    # Split genome file into multiple separate files  # Z: each 'genome'
    for genome in SeqIO.parse(GENOMES_FILE2, format="fasta"):
        filepath = genome.description.replace(" ", "&").replace("/", "&").replace(",", "&")
        filepath += ".fasta"
        SeqIO.write(genome, join(GENOMES_FOLDER, filepath), format="fasta")

    # STEP 2: Simulate Amplicon Population
    genome_counter = 0
    lineages_formatted = []
    for genome_path in genome_abundances:
        genome_counter += 1
        genome_path = genome_path.replace(" ", "_").replace("/", "&").replace(",", "&") + ".fasta"
        lineages_formatted.append(genome_path[:-6])
        genome_path = join(GENOMES_FOLDER, genome_path)
        genome_filename_short = ".".join(basename(genome_path).split(".")[:-1])
        lineage_reference = SeqIO.read(genome_path, format="fasta")

        # use BWA to create a dataframe with positions of each primer pair aligned to the genome
        if VERBOSE:
            logging.info(f"Working on genome {genome_counter} of {n_genomes}")
            logging.info(f"Using BWA to align primers to genome {lineage_reference.description}")

        hp.build_index(genome_path, join(INDICES_FOLDER, genome_filename_short))
        df = get_alignment_df_and_call_SNVs(
            genome_path, genome_filename_short, INDICES_FOLDER, PRIMERS_FILE, TEMP_FOLDER, VERBOSE
        )
        df["abundance"] = genome_abundances[df["ref"][0]]

        # write the amplicons to a file
        write_amplicon(df, lineage_reference, genome_filename_short, AMPLICONS_FOLDER)

        amplicon_df = pd.concat([amplicon_df, df])

    # Set amplicon distribution based on hyperparameters
    amplicon_df.set_index(["amplicon_number", "alt_num_left", "alt_num_right"], inplace=True)
    # amplicon_df.reset_index(drop=True, inplace=True)
    amp_dist_df = load_amp_dist_file(AMPLICON_DISTRIBUTION_FILE, PRIMER_BED)
    amplicon_df = amplicon_df.merge(
        amp_dist_df,
        left_index=True,
        right_on=["amplicon_number", "alt_num_left", "alt_num_right"],
        how="left"
    )

    # STEP 3: Library Prep - PCR Amplification of Amplicons
    # This will distribute the "hyperparameter" across different versions (i.e. different errors)
    # of the same amplicon using the VAF parameters. So hyperparam(original_amplicon) = hyperparam(amp_v1) + hyperparam(amp_v2)
    if not NO_PCR_ERRORS:
        if VERBOSE:
            logging.info(f"Adding high-frequency errors")

        PATHS = {
            "PRIMER_BED": PRIMER_BED,
            "REFERENCE": REFERENCE,
            "INDEX_BASE": INDEX_BASE,
            "AMPLICONS_FOLDER": AMPLICONS_FOLDER
        }

        # Kind of by definition, you can't know which errors come from the PCR process and which from the wastewater
        # environment. These two together are the "high-frequency errors" and encompasses both
        amplicon_df, vcf_errordf = add_high_frequency_errors(
            amplicon_df, genome_abundances, PATHS, REF_NAME, RATES, DEL_LENGTH_GEOMETRIC_PARAMETER,
            INS_MAX_LENGTH, VAF_PARAMETER_DICT, R_VAF_PARAMETER_DICT, DISALLOWED_POSITIONS, RNG
        )

        if isinstance(vcf_errordf, str) and VERBOSE:
            logging.info(f"No high-frequency errors were introduced! Possible reason: too low error rates.")
        else:
            vcf_path = f"{OUTPUT_FOLDER}/{OUTPUT_FILENAME_PREFIX}_hf_errors.vcf"
            if os.path.exists(vcf_path):
                os.remove(vcf_path)
            with open(vcf_path, "w") as o:
                o.writelines([
                    "##fileformat=VCFv4.3\n",
                    f"##reference={REF_NAME}\n",
                    f'##contig=<ID={REF_NAME},length={REF_LEN}>\n'
                    '##INFO=<ID=VAF,Number=A,Type=Float,Description="Variant Allele Frequency">\n',
                    '##INFO=<ID=REC,Number=A,Type=String,Description="Recurrence state across source genomes. R: recurrent; U: unique to genome">\n',
                    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
                ])

            vcf_errordf.to_csv(vcf_path, mode="a", header=False, index=False, sep="\t", float_format='%.5f')
            if VERBOSE:
                logging.info(
                    f'All aimed high-frequency errrors are written to "{OUTPUT_FOLDER}/{OUTPUT_FILENAME_PREFIX}_hf_errors.vcf"')

    # Now that we have each variation of amplicon and their corresponding 'hyperparameter',
    # we can add the SNV-bias based on it and see how many reads of each ART has to generate.
    if VERBOSE:
        logging.info(f"Adding SNV bias (balance={SNV_BALANCE})")
    amplicon_df = apply_bias(
        amplicon_df, TEMP_FOLDER, RNG, N_READS,
        genome_abundances, AMPLICON_DIRICHLET_PARAMETER,
        SNV_DIRICHLET_PARAMETER, SNV_BALANCE
    )
    # amplicon_df = correct_dropout_rate(amplicon_df, DROPOUT_RATE, RNG)

    # pick total numbers of reads for each amplicon
    # df_amplicons = apply_amplicon_reads_sampler(
    #     df_amplicons,
    #     AMPLICON_DISTRIBUTION,
    #     amp_dist_df,
    #     AMPLICON_DIRICHLET_PARAMETER,
    #     genome_abundances,
    #     N_READS,
    #     RNG
    # )
    # df_amplicons.reset_index(drop=True, inplace=True)
    if VERBOSE:
        logging.info(f"Total number of reads was {sum(amplicon_df['n_reads'])}, when {N_READS} was expected.")

    # write a summary csv
    amplicon_df[[
        "ref", "amplicon_number", "alt_num_left", "alt_num_right", "var_num",
        "total_n_reads", "abundance", "genome_n_reads",
        "hyperparameter", "amplicon_prop", "SNVs_in_primers", "n_reads"
    ]].to_csv(join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME_PREFIX}_amplicon_abundances_summary.tsv"), sep="\t")

    # STEP 4: Simulate Reads
    filepaths, n_reads = format_ART_input(
        lineages_formatted, amplicon_df["amplicon_filepath"].tolist(),
        amplicon_df["n_reads"].tolist(), AMPLICONS_FOLDER
    )

    t2 = perf_counter()
    t_pre_art = t2 - t1

    logging.info("Generating reads using art_illumina, cycling through all genomes and remaining amplicons.")
    with art_illumina(
        OUTPUT_FOLDER, OUTPUT_FILENAME_PREFIX,
        READ_LENGTH, SEQ_SYS, QPROF1, QPROF2,
        VERBOSE, TEMP_FOLDER, N_READS, FRAGMENT_AMPLICONS,
        FRAGMENT_LEN_MEAN, FRAGMENT_LEN_SD, ART_QSHIFT
    ) as art:
        art.run(filepaths, n_reads, RNG)
    t_art = perf_counter() - t2

    # STEP 5: Clean up all of the temp. directories
    if AUTOREMOVE:
        remove_tmp(TEMP_FOLDER)
    if VERBOSE:
        logging.info(f"Time (pre-ART): {t_pre_art:.3f}")
        logging.info(f"Time (ART)    : {t_art:.3f}")
        logging.info(f"Time (Total)  : {perf_counter() - t1:.3f}")
