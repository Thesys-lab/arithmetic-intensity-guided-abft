import argparse

parser = argparse.ArgumentParser()
parser.add_argument("m", type=int)
parser.add_argument("n", type=int)
parser.add_argument("k", type=int)
parser.add_argument("prec", type=int)
args = parser.parse_args()

ops = 2 * args.m * args.n * args.k
elts = (args.m * args.n) + (args.m * args.k) + (args.k * args.n)
print("{:.2f}".format(ops / (elts * args.prec)))
