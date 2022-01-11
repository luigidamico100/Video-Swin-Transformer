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

EXPERIMENT_NAME=22

work_dir_path = '/home/luigi.damico/Video-Swin-Transformer/work_dirs/experiment_'+str(EXPERIMENT_NAME)+'/' if on_cuda else '/Users/luigidamico/Documents/GitHub/Video-Swin-Transformer/work_dirs/experiment_'+str(EXPERIMENT_NAME)+'/'
# dataset_path = '/home/luigi.damico/ICPR_datasets/ICPR_rawframes_numframes2/' if on_cuda else '/Volumes/SD Card/Thesis/ICPR/Dataset_rawframe/ICPR_rawframes_numframes2/'

def plot_training(training_stats_df, testfold):
    fig, axs = plt.subplots(2,1)
    training_stats_df.loc[testfold][['Loss_train','Loss_val','Loss_test']].plot(ax=axs[0])
    training_stats_df.loc[testfold][['Accuracy_train','Accuracy_val','Accuracy_test']].plot(ax=axs[1])
    for ax in axs: ax.grid()
    fig.suptitle(f'test fold: {testfold}')
    
    
#%% reading files
training_stats_df = pd.read_csv(work_dir_path + 'results/results_stats.csv', index_col=[0,1])

with open(work_dir_path + 'results/best_epoch_dict.pkl', 'rb') as handle:   best_epochs_dict = pickle.load(handle)


#%% main
for testfold in range(0, 10): plot_training(training_stats_df, testfold=testfold)






