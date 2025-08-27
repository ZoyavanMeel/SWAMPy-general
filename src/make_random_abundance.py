import pandas as pd
import random
import argparse
import sys


def change_abundance(abundance_file: str, clinical: str, output_file: str, seed: int) -> None:
    random.seed(seed)
    df = pd.read_csv(abundance_file, header=None, sep="\t", index_col=None, names=["genome", "abu"])

    if clinical == "clinical":
        g = random.choice(df["genome"])
        with open(output_file, "w") as fh:
            fh.write(f"{g}\t1.0")
        return

    elif clinical == "wastewater":
        raw_abus = [random.randint(0, 100) for _ in range(df.shape[0])]
        norm = sum(raw_abus)
        df["abu"] = [abu/norm for abu in raw_abus]
        df.to_csv(output_file, sep="\t", index=False, header=False)
        return
    else:
        raise ValueError(f"'-c' must be 'clinical' or 'wastewater', but was: {clinical}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--abundance_file", "-a")
    parser.add_argument("--clinical", "-c")
    parser.add_argument("--output_file", "-o")
    parser.add_argument("--seed", "-s")
    args = parser.parse_args()

    change_abundance(args.abundance_file, args.clinical, args.output_file, int(args.seed))
