from DataUtils import *
from graph_preprocess import *
import sys, os


dataset = None
if int(sys.argv[1]) == 0:
    dataset = 'davis'
else:
    dataset = 'kiba'

generate_pytorch_data([dataset], 1, dataset)
