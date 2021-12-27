#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:06:56 2021

@author: luigidamico
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import torch
import json
from collections import defaultdict
import argparse

on_cuda = torch.cuda.is_available()


def get_training_data(work_dir_path, num_epochs, num_folds=10):
    
    def load_json_logs(json_logs):
        # load and convert json_logs to log_dict, key is epoch, value is a sub dict
        # keys of sub dict is different metrics, e.g. memory, top1_acc
        # value of sub dict is a list of corresponding values of all iterations
        log_dicts = [dict() for _ in json_logs]
        for json_log, log_dict in zip(json_logs, log_dicts):
            with open(json_log, 'r') as log_file:
                for line in log_file:
                    log = json.loads(line.strip())
                    # skip lines without `epoch` field
                    if 'epoch' not in log:
                        continue
                    epoch = log.pop('epoch')
                    if epoch not in log_dict:
                        log_dict[epoch] = defaultdict(list)
                    for k, v in log.items():
                        log_dict[epoch][k].append(v)
        return log_dicts

    idx = pd.MultiIndex.from_product([range(0,num_folds), range(1, num_epochs+1)], names=['testfold', 'epoch'])
    dataframe_stats = pd.DataFrame(columns=['Loss_train', 'Accuracy_train'], index=idx)
 
    for testfold in range(0,num_folds):
        train_data_path = glob.glob(work_dir_path + 'training/testfold_' + str(testfold) + '/*.json')
        log_dicts = load_json_logs(train_data_path)
        for epoch, value in log_dicts[0].items():
            dataframe_stats.loc[testfold, epoch] = [sum(value['loss'])/len(value['loss']), sum(value['top1_acc'])/len(value['top1_acc'])]

    return dataframe_stats


def get_evaluation_data(work_dir_path, dataset_dir_path, num_epochs, num_folds=10, phase='val'):
    
    idx = pd.MultiIndex.from_product([range(0,num_folds), range(1, num_epochs+1)], names=['testfold', 'epoch'])
    dataframe_stats = pd.DataFrame(columns=['Loss_'+phase, 'Accuracy_'+phase], index=idx)
 
    for testfold in range(0,num_folds):
        dataset = pd.read_csv(dataset_dir_path + 'foldtest_' + str(testfold) + '/ICPR_' + phase + '_list_informations.csv')
        true_labels = np.array(dataset['classe'] == 'RDS').astype(int)
        
        for epoch in range(1, num_epochs+1):
            prediction_path = work_dir_path + 'predictions/' + phase + '/testfold_' + str(testfold) + '/epoch_' + str(epoch) + '.pkl'
            with open(prediction_path, 'rb') as f:
                prediction = np.array(pickle.load(f))
            pred_labels = prediction.argmax(axis=1)
            dataframe_stats.loc[testfold, epoch]['Accuracy_'+phase] = accuracy_score(true_labels, pred_labels)
            loss = torch.nn.CrossEntropyLoss()
            dataframe_stats.loc[testfold, epoch]['Loss_'+phase] = loss(torch.tensor(prediction), torch.tensor(true_labels)).item()
    
    return dataframe_stats


def get_information_data(work_dir_path, dataset_dir_path, best_epoch_dict, num_folds=10, phase='val'):
    dataset_final = pd.DataFrame()
    
    for testfold in range(0, num_folds):
        dataset_information = pd.read_csv(dataset_dir_path + 'foldtest_' + str(testfold) + '/ICPR_'  + phase + '_list_informations.csv', index_col=0)

        prediction_path = work_dir_path + 'predictions/' + phase + '/testfold_' + str(testfold) + '/epoch_' + str(best_epoch_dict[testfold]) + '.pkl'
        with open(prediction_path, 'rb') as f:
            prediction = np.array(pickle.load(f))
        dataset_information['testfold'] = testfold
        dataset_information[['BEST_prob', 'RDS_prob']] = prediction
        dataset_information['predicted_class'] = prediction.argmax(axis=1)
        dataset_information['true_class'] = (dataset_information['classe'] == 'RDS').astype(int)
        dataset_final = dataset_final.append(dataset_information)

    return dataset_final


def find_best_model(dataframe_eval, work_dir_path, num_folds, keep_only_best=False):

    def keep_only_best_fun(epoch_to_be_kept, dir_path):
        for model_pth in glob.glob(dir_path + '*.pth'):
            if model_pth.split('/')[-1] != 'epoch_' + str(epoch_to_be_kept) + '.pth':
                os.remove(model_pth)

    best_epoch_dict = {}
    for testfold in range(0, num_folds):
        best_epoch = pd.to_numeric(dataframe_eval.loc[testfold]['Loss_val']).idxmin()
        best_epoch_dict[testfold] = best_epoch
        if keep_only_best:
            dir_path = work_dir_path + 'training/testfold_' + str(testfold) + '/'
            keep_only_best_fun(epoch_to_be_kept=best_epoch, dir_path=dir_path)
    return best_epoch_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Generate prediction results .csv files')
    parser.add_argument('work_dir_path', help='working directory')
    parser.add_argument('dataset_dir_path', help='dataset directory')
    parser.add_argument('num_epochs', help='number of epochs')
    parser.add_argument('num_folds', help='number of folds')
    parser.add_argument('results_path', help='path of csv results')
    parser.add_argument('keep_only_best', help='decide if only the best model has to be kept')
    args = parser.parse_args()
    return args

#%%
    
# work_dir_path = '/home/luigi.damico/Video-Swin-Transformer/work_dirs/experiment_10/' if on_cuda else '/Users/luigidamico/Documents/GitHub/Video-Swin-Transformer/work_dirs/experiment_10/'
# dataset_dir_path = '/home/luigi.damico/ICPR_datasets/ICPR_rawframes_numframes2/' if on_cuda else '/Volumes/SD Card/Thesis/ICPR/Dataset_rawframe/ICPR_rawframes_numframes2/'
# num_epochs = 20
# num_folds = 10
# keep_only_best = 0


if __name__ == '__main__':
    
    args = parse_args()
    work_dir_path = args.work_dir_path
    dataset_dir_path = args.dataset_dir_path
    num_epochs = int(args.num_epochs)
    num_folds = int(args.num_folds)
    results_path = args.results_path
    keep_only_best = int(args.keep_only_best)
    #keep_only_best = 0
    
    train_dataframe_eval = get_training_data(work_dir_path, num_epochs, num_folds=num_folds)
    val_dataframe_eval = get_evaluation_data(work_dir_path, dataset_dir_path, num_epochs, num_folds=num_folds, phase='val')
    test_dataframe_eval = get_evaluation_data(work_dir_path, dataset_dir_path, num_epochs, num_folds=num_folds, phase='test')
    
    dataframe_eval = pd.merge(train_dataframe_eval, val_dataframe_eval, left_index=True, right_index=True)
    dataframe_eval = pd.merge(dataframe_eval, test_dataframe_eval, left_index=True, right_index=True)

    best_epoch_dict = find_best_model(dataframe_eval, work_dir_path, num_folds, keep_only_best=keep_only_best)

    test_dataframe_information = get_information_data(work_dir_path, dataset_dir_path, best_epoch_dict, num_folds=num_folds, phase='test')

    os.mkdir(results_path)
    dataframe_eval.to_csv(results_path + 'results_stats.csv')
    test_dataframe_information.to_csv(results_path + 'results_information.csv')
    pickle.dump(best_epoch_dict, open(results_path + "best_epoch_dict.pkl", "wb"))
