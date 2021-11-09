import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("ogfile", type=str,
                    help="CSV file containing original results")
parser.add_argument("repfile", type=str,
                    help="CSV file containing replication results")
parser.add_argument("outfile", type=str,
                    help="CSV file to write to")
args = parser.parse_args()


df_og = pd.read_csv(args.ogfile)
df_rep = pd.read_csv(args.repfile)
overall = pd.merge(df_og, df_rep, on="Operation")
overall["Rep_Overhead"] = overall["Runtime_y"] / overall["Runtime_x"]
#print(overall["Rep_Overhead"].mean())
overall.to_csv(args.outfile, index=False)

min_og = df_og["Runtime"].min()
min_rep = df_rep["Runtime"].min()
#print(min_og)
#print(min_rep)
print(min_rep / min_og)
