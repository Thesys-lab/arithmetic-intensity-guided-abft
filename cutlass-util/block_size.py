import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("--mode", type=str, default="blocks_m",
                    choices=["blocks_m", "blocks_for_block_m"])
parser.add_argument("--block_m_to_query", type=int)
args = parser.parse_args()

df = pd.read_csv(args.infile)

df["blocks_m"] = np.ceil(df["m"] / df["cta_m"])
df["blocks_n"] = np.ceil(df["n"] / df["cta_n"])
df["total_blocks"] = df["blocks_m"] * df["blocks_n"]

if args.mode == "blocks_m":
    for val in df["blocks_m"].unique():
        print(int(val))
elif args.mode == "blocks_for_block_m":
    for val in df.loc[df["blocks_m"] == args.block_m_to_query]["total_blocks"].unique():
        print(int(val))
