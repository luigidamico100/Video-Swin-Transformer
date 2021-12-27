#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:54:07 2021

@author: luigidamico
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

on_cuda = torch.cuda.is_available()

work_dir_path = '/home/luigi.damico/Video-Swin-Transformer/work_dirs/experiment_2/' if on_cuda else '/Users/luigidamico/Documents/GitHub/Video-Swin-Transformer/work_dirs/experiment_2/'
dataset_path = '/home/luigi.damico/ICPR_datasets/ICPR_rawframes_numframes2/' if on_cuda else '/Volumes/SD Card/Thesis/ICPR/Dataset_rawframe/ICPR_rawframes_numframes2/'


#%%
dataframe_stats = pd.read_csv(work_dir_path + 'results/results_stats.csv', index_col=[0,1])

with open(work_dir_path + 'results/best_epoch_dict.pkl', 'rb') as handle:
    best_epochs_dict = pickle.load(handle)


testfold = 2
fig, axs = plt.subplots(2,1)
dataframe_stats.loc[testfold][['Loss_train','Loss_val','Loss_test']].plot(ax=axs[0])
dataframe_stats.loc[testfold][['Accuracy_train','Accuracy_val','Accuracy_test']].plot(ax=axs[1])


test_informations = pd.read_csv(work_dir_path + 'results/results_information.csv', index_col=0)

accuracy_score(test_informations['true_class'], test_information['predicted_class'])