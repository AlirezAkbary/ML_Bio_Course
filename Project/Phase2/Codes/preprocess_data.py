from DataUtils import *
from graph_preprocess import *
import sys, os

datasets = sys.argv[1]
if datasets == "all":
    datasets = ['davis', 'kiba']
else:
    if int(datasets) == 0:
        datasets = ['davis']
    else:
        datasets = ['kiba']

generate_csvs(datasets)
generate_pytorch_data(datasets)