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
from sklearn.metrics import accuracy_score, confusion_matrix
import glob
import torch
import json
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
np.set_printoptions(4)

on_cuda = torch.cuda.is_available()


def get_training_data(work_dir_path, num_folds=10):
    
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

    num_epochs =len(load_json_logs(glob.glob(work_dir_path + 'training/testfold_0/*.json'))[0])
    idx = pd.MultiIndex.from_product([range(0,num_folds), range(1, num_epochs+1)], names=['testfold', 'epoch'])
    dataframe_stats = pd.DataFrame(columns=['Loss_train', 'Accuracy_train'], index=idx)
 
    for testfold in range(0,num_folds):
        train_data_path = glob.glob(work_dir_path + 'training/testfold_' + str(testfold) + '/*.json')
        log_dicts = load_json_logs(train_data_path)
        for epoch, value in log_dicts[0].items():
            dataframe_stats.loc[testfold, epoch] = [sum(value['loss'])/len(value['loss']), sum(value['top1_acc'])/len(value['top1_acc'])]
    return dataframe_stats, num_epochs


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


def get_training_history_df(work_dir_path, dataset_dir_path, num_folds):
    train_dataframe_eval, num_epochs = get_training_data(work_dir_path, num_folds=num_folds)
    val_dataframe_eval = get_evaluation_data(work_dir_path, dataset_dir_path, num_epochs, num_folds=num_folds,
                                             phase='val')
    test_dataframe_eval = get_evaluation_data(work_dir_path, dataset_dir_path, num_epochs, num_folds=num_folds,
                                              phase='test')

    training_history_df = pd.merge(train_dataframe_eval, val_dataframe_eval, left_index=True, right_index=True)
    training_history_df = pd.merge(training_history_df, test_dataframe_eval, left_index=True, right_index=True)

    return training_history_df, num_epochs


def get_predictions_df(work_dir_path, dataset_dir_path, best_epochs_dict, num_folds=10, phase='val'):
    dataset_final = pd.DataFrame()

    for testfold in range(0, num_folds):
        dataset_information = pd.read_csv(dataset_dir_path + 'foldtest_' + str(testfold) + '/ICPR_'  + phase + '_list_informations.csv', index_col=0)

        prediction_path = work_dir_path + 'predictions/' + phase + '/testfold_' + str(testfold) + '/epoch_' + str(best_epochs_dict[testfold]) + '.pkl'
        with open(prediction_path, 'rb') as f:
            prediction = np.array(pickle.load(f))
        dataset_information['testfold'] = testfold
        dataset_information[['BEST_prob', 'RDS_prob']] = prediction
        dataset_information['predicted_class'] = prediction.argmax(axis=1)
        dataset_information['true_class'] = (dataset_information['classe'] == 'RDS').astype(int)
        dataset_final = dataset_final.append(dataset_information)
        
    dataset_final['ospedale'] = create_ospedale_column(dataset_final)

    return dataset_final


def find_best_model(training_history_df, work_dir_path, num_folds, keep_only_best=False):

    def keep_only_best_fun(epoch_to_be_kept, dir_path):
        for model_pth in glob.glob(dir_path + '*.pth'):
            if model_pth.split('/')[-1] != 'epoch_' + str(epoch_to_be_kept) + '.pth':
                os.remove(model_pth)

    best_epochs_dict = {}
    for testfold in range(0, num_folds):
        best_epoch = pd.to_numeric(training_history_df.loc[testfold]['Loss_val']).idxmin()
        best_epochs_dict[testfold] = best_epoch
        if keep_only_best:
            dir_path = work_dir_path + 'training/testfold_' + str(testfold) + '/'
            keep_only_best_fun(epoch_to_be_kept=best_epoch, dir_path=dir_path)
    return best_epochs_dict


def get_best_epochs_df(training_history_df, best_epochs_dict):
    training_history_df = training_history_df.copy()
    results_df = pd.DataFrame(columns=['Best_epoch', 'Loss_val', 'Accuracy_test'])
    results_df.index.name = 'testfold'
    training_history_df['Best_epoch'] = training_history_df.index.get_level_values('epoch')
    for testfold in training_history_df.index.get_level_values('testfold').unique():
        results_df.loc[testfold] = training_history_df.loc[testfold, best_epochs_dict[testfold]][['Best_epoch', 'Loss_val', 'Accuracy_test']]
    return results_df


def get_metrics_df(predictions_df):
    accuracy = accuracy_score(predictions_df['true_class'], predictions_df['predicted_class'])
    conf_matrix = confusion_matrix(predictions_df['true_class'], predictions_df['predicted_class'])
    conf_matrix_norm = confusion_matrix(predictions_df['true_class'], predictions_df['predicted_class'], normalize='true')
    spearman_corr = spearmanr(predictions_df['BEST_prob'], predictions_df['emogas_index'])[0]
    metrics_list = [accuracy, conf_matrix, conf_matrix_norm, spearman_corr]
    return np.array(metrics_list, dtype=object)


def get_results_df_hospital(predictions_df):
    predictions_df_naples = predictions_df[predictions_df['ospedale']=='Naples']
    predictions_df_florence = predictions_df[predictions_df['ospedale']=='Florence']
    predictions_df_milan = predictions_df[predictions_df['ospedale']=='Milan']
    results_df_hospital = pd.DataFrame(columns=['Accuracy', 'Conf_matrix', 'Conf_matrix_norm', 'Spearman_corr'])
    results_df_hospital.loc['Naples'] = get_metrics_df(predictions_df_naples)
    results_df_hospital.loc['Florence'] = get_metrics_df(predictions_df_florence)
    results_df_hospital.loc['Milan'] = get_metrics_df(predictions_df_milan)
    results_df_hospital.loc['Overall'] = get_metrics_df(predictions_df)
    return results_df_hospital
    

def get_results_df_nWrongPatients(predictions_df):
    
    predictions_df_wrong = predictions_df[predictions_df['true_class']!=predictions_df['predicted_class']]
    predictions_df_wrong_grupbyPatients = predictions_df_wrong.groupby(by=['bimbo_name', 'classe', 'ospedale']).count()     # could be also .mean() or others...
    predictions_df_wrong_grupbyPatients['n_patients_misclassfied'] = 1
    predictions_df_wrong_grupbyHospClass = predictions_df_wrong_grupbyPatients.groupby(['classe', 'ospedale']).count()
    
    return predictions_df_wrong_grupbyHospClass['n_patients_misclassfied']
    

def create_ospedale_column(dataset):
    ospedale_col = pd.Series(index=dataset.index, dtype=object)
    for idx in dataset.index:
        bimbo_name = dataset.loc[idx]['bimbo_name']
        if 'Buzzi' in bimbo_name:
            ospedale_col.loc[idx] = 'Milan'
        elif 'Firenze' in bimbo_name:
            ospedale_col.loc[idx] = 'Florence'
        else:
            ospedale_col.loc[idx] = 'Naples'
    return ospedale_col


def plot_training(training_stats_df, testfold, outfile_path=None):
    fig, axs = plt.subplots(2,1)
    training_stats_df.loc[testfold][['Loss_train','Loss_val','Loss_test']].plot(ax=axs[0])
    training_stats_df.loc[testfold][['Accuracy_train','Accuracy_val','Accuracy_test']].plot(ax=axs[1])
    for ax in axs: ax.grid()
    fig.suptitle(f'test fold: {testfold}')
    if outfile_path: fig.savefig(outfile_path)


def save_results_to_file(results_path, training_history_df, val_predictions_df, test_predictions_df, 
                         best_epochs_dict, val_results_df_hospital, test_results_df_hospital, test_results_df_nWrongPatients):
    os.mkdir(results_path)
    os.mkdir(results_path+'training_history_plots')
    training_history_df.to_csv(results_path + 'training_history_df.csv', float_format='%.4f')
    val_predictions_df.to_csv(results_path + 'val_predictions_df.csv', float_format='%.4f')
    test_predictions_df.to_csv(results_path + 'test_predictions_df.csv', float_format='%.4f')
    pickle.dump(best_epochs_dict, open(results_path + "best_epoch_dict.pkl", "wb"))
    best_epochs_df.to_csv(results_path + 'best_epochs_df.csv', float_format='%.4f')
    val_results_df_hospital.to_csv(results_path + 'val_results_df_hospital.csv', float_format='%.4f')
    test_results_df_hospital.to_csv(results_path + 'test_results_df_hospital.csv', float_format='%.4f')
    test_results_df_nWrongPatients.to_csv(results_path + 'test_results_df_nWrongPatients.csv', float_format='%.4f')
    for testfold in range(0, 10): plot_training(training_history_df, testfold=testfold, outfile_path=results_path+'training_history_plots/Test fold '+str(testfold)+'.jpg')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate prediction results .csv files')
    parser.add_argument('work_dir_path', help='working directory')
    parser.add_argument('dataset_dir_path', help='dataset directory')
    parser.add_argument('num_folds', help='number of folds')
    parser.add_argument('results_path', help='path of csv results')
    parser.add_argument('keep_only_best', help='decide if only the best model has to be kept')
    args = parser.parse_args()
    return args

#%%
    
work_dir_path = '/home/luigi.damico/Video-Swin-Transformer/work_dirs_swin/experiment_2.1/' if on_cuda else '/Users/luigidamico/Documents/GitHub/work_dirs_swin/experiment_2.1/'
dataset_dir_path = '/home/luigi.damico/ICPR_datasets/ICPR_allFrames/' if on_cuda else '/Volumes/SD Card/Thesis/ICPR/Dataset_rawframe/ICPR_allFrames/'
results_path = work_dir_path + 'results/'
num_folds = 10
keep_only_best = 0


#%%

if __name__ == '__main__':
    
    args = parse_args()
    work_dir_path = args.work_dir_path
    dataset_dir_path = args.dataset_dir_path
    num_folds = int(args.num_folds)
    results_path = args.results_path
    keep_only_best = int(args.keep_only_best)
    
    training_history_df, num_epochs = get_training_history_df(work_dir_path, dataset_dir_path, num_folds)
    best_epochs_dict = find_best_model(training_history_df, work_dir_path, num_folds, keep_only_best=keep_only_best)
    val_predictions_df = get_predictions_df(work_dir_path, dataset_dir_path, best_epochs_dict, num_folds=num_folds, phase='val')
    test_predictions_df = get_predictions_df(work_dir_path, dataset_dir_path, best_epochs_dict, num_folds=num_folds, phase='test')
    best_epochs_df = get_best_epochs_df(training_history_df, best_epochs_dict)
    val_results_df_hospital = get_results_df_hospital(val_predictions_df)
    test_results_df_hospital = get_results_df_hospital(test_predictions_df)
    test_results_df_nWrongPatients = get_results_df_nWrongPatients(test_predictions_df)
    
    save_results_to_file(results_path, training_history_df, val_predictions_df, test_predictions_df,
                         best_epochs_dict, val_results_df_hospital, test_results_df_hospital, test_results_df_nWrongPatients)

