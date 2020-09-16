import os
import numpy as np
import pandas as pd
import json, pickle
from collections import OrderedDict
from tqdm.auto import tqdm
from rdkit import Chem
from graph_preprocess import *
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import networkx as nx
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
import os
from sklearn.utils import resample
import sys, os

all_prots = []


def generate_csvs(datasets):
    for dataset in datasets:
        print('convert data from DeepDTA for ', dataset)
        fpath = 'data/' + dataset + '/'
        whole_train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        train_fold = [ee for e in range(len(whole_train_fold) - 1) for ee in whole_train_fold[e]]
        validation_fold = [ee for ee in whole_train_fold[-1]]
        test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
        drugs = []
        prots = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            drugs.append(lg)
        for t in proteins.keys():
            prots.append(proteins[t])
        if dataset == 'davis':
            affinity = [-np.log10(y / 1e9) for y in affinity]
        affinity = np.asarray(affinity)
        opts = ['train', 'test', 'validation']
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity) == False)
            if opt == 'train':
                rows, cols = rows[train_fold], cols[train_fold]
            elif opt == 'test':
                rows, cols = rows[test_fold], cols[test_fold]
            elif opt == 'validation':
                rows, cols = rows[validation_fold], cols[validation_fold]
            with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [drugs[rows[pair_ind]]]
                    ls += [prots[cols[pair_ind]]]
                    ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                    f.write(','.join(map(str, ls)) + '\n')
            # if opt == 'train':
            #     train_csv = pd.read_csv('data/' + dataset + '_' + opt + '.csv')
            #     large = train_csv[train_csv.affinity <= 7]
            #     small = train_csv[train_csv.affinity > 7]
            #     small_upsampled = resample(small, replace=True, n_samples=9 * len(large) // 10, random_state=1)
            #     balanced_train =  pd.concat([small_upsampled, large]).sample(frac=1)
            #     balanced_train.to_csv('data/' + dataset + '_' + 'balanced_' + opt + '.csv')

        print("train, validation, test for ",dataset, "created.")


def generate_pytorch_data(datasets, mode = 0, my_dataset='davis'):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    seq_dict_len = len(seq_dict)
    max_seq_len = 1000

    compound_iso_smiles = []
    for dt_name in datasets:
        opts = ['train', 'test', 'validation']
        for opt in opts:
            df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
            compound_iso_smiles += list(df['compound_iso_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    if mode == 0:

    # convert to PyTorch data format
        for dataset in datasets:
            processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
            processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
            if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):

                df = pd.read_csv('data/' + dataset + '_train.csv')
                train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                    df['affinity'])
                XT = [seq_cat(t) for t in train_prots]
                train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)


                df = pd.read_csv('data/' + dataset + '_test.csv')
                test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                    df['affinity'])
                XT = [seq_cat(t) for t in test_prots]
                test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

                df = pd.read_csv('data/' + dataset + '_validation.csv')
                validation_drugs, validation_prots, validation_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                    df['affinity'])
                XT = [seq_cat(t) for t in validation_prots]
                validation_drugs, validation_prots, validation_Y = np.asarray(validation_drugs), np.asarray(XT), np.asarray(validation_Y)

                # df = pd.read_csv('data/' + dataset + '_balanced_train.csv')
                # balanced_train_drugs, balanced_train_prots, balanced_train_Y = list(df['compound_iso_smiles']), list(
                #     df['target_sequence']), list(
                #     df['affinity'])
                # XT = [seq_cat(t) for t in balanced_train_prots]
                # balanced_train_drugs, balanced_train_prots, balanced_train_Y = np.asarray(balanced_train_drugs), np.asarray(XT), np.asarray(
                #     balanced_train_Y)

                # make data PyTorch Geometric ready
                print('preparing ', dataset + '_train.pt in pytorch format!')
                train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots, y=train_Y,
                                            smile_graph=smile_graph)
                print('preparing ', dataset + '_test.pt in pytorch format!')
                test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots, y=test_Y,
                                           smile_graph=smile_graph)

                print('preparing ', dataset + '_validation.pt in pytorch format!')
                validation_data = TestbedDataset(root='data', dataset=dataset + '_validation', xd=validation_drugs, xt=validation_prots, y=validation_Y,
                                           smile_graph=smile_graph)

                # print('preparing ', dataset + '_balanced_train.pt in pytorch format!')
                # balanced_train_data = TestbedDataset(root='data', dataset=dataset + '_balanced_train', xd=balanced_train_drugs, xt=balanced_train_prots, y=balanced_train_Y,
                #                            smile_graph=smile_graph)

            else:
                print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')

    else:

        opt = 'train'
        train_csv = pd.read_csv('data/' + my_dataset + '_' + opt + '.csv')
        if my_dataset == 'davis':
            large = train_csv[train_csv.affinity <= 7]
            small = train_csv[train_csv.affinity > 7]
        else:
            large = train_csv[train_csv.affinity <= 12.1]
            small = train_csv[train_csv.affinity > 12.1]
        small_upsampled = resample(small, replace=True, n_samples=9 * len(large) // 10, random_state=1)
        balanced_train = pd.concat([small_upsampled, large]).sample(frac=1)
        balanced_train.to_csv('data/' + my_dataset + '_' + 'balanced_' + opt + '.csv')

        df = pd.read_csv('data/' + my_dataset + '_balanced_train.csv')
        balanced_train_drugs, balanced_train_prots, balanced_train_Y = list(df['compound_iso_smiles']), list(
            df['target_sequence']), list(
            df['affinity'])
        XT = [seq_cat(t) for t in balanced_train_prots]
        balanced_train_drugs, balanced_train_prots, balanced_train_Y = np.asarray(balanced_train_drugs), np.asarray(
            XT), np.asarray(
            balanced_train_Y)

        print('preparing ', my_dataset + '_balanced_train.pt in pytorch format!')
        balanced_train_data = TestbedDataset(root='data', dataset=my_dataset + '_balanced_train',
                                                xd=balanced_train_drugs, xt=balanced_train_prots, y=balanced_train_Y,
                                                smile_graph=smile_graph)


