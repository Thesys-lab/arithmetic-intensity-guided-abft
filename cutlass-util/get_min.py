import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str,
                    help="CSV file containing results")
args = parser.parse_args()


df_og = pd.read_csv(args.infile)
min_og = df_og["Runtime"].min()
print(min_og)
