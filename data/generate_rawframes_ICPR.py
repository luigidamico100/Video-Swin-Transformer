#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:47:42 2021

@author: luigidamico
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
from random import choice
import cv2
import os
import torch
import random
NUM_ROWS = 224

on_cuda = torch.cuda.is_available()


def default_mat_loader(path, num_frames=2, verbose=False):

    matdata = loadmat(path)
    data = []
    frame_keys = [k for k in matdata.keys() if k.startswith('f') and len(k)<3]
    startframe_idx = random.randint(0, len(frame_keys)-num_frames)
    selected_frame_keys = frame_keys[startframe_idx:startframe_idx+num_frames]
    if verbose:
        print(path)
        print(frame_keys) 
        print(selected_frame_keys)
        print('-'*10)
    for k in selected_frame_keys:
            data.append(matdata[k][:NUM_ROWS])
    data = np.array(data)

    processed_video_path = path.split('/')
    information_dict = {
            'bimbo_name': str(matdata['bimbo_name'][0]),
            'classe': str(matdata['classe'][0]),
            'esame_name': str(matdata['esame_name'][0]),
            'paziente': str(matdata['paziente'][0][0]),
            'emogas_index': matdata['valore'][0][0] / 480.,
            'video_name': str(matdata['video_name'][0]),
            'processed_video_name': processed_video_path[-2] + '/' + processed_video_path[-1],
            'frame_key': str(selected_frame_keys), 
            'total_clip_frames': len(frame_keys)
        }
    
    return data, information_dict
    
    
#%%

dataset_test = pd.read_csv('/home/luigi.damico/DL_QLUS/Experiments/experiment_allfold_exp_0/evaluation_dataframe.csv' if on_cuda else'/Volumes/SD Card/Thesis/Experiments/models_training/experiment_allfold_exp_0/evaluation_dataframe.csv', index_col=0)
dataset_test.drop(index='RDS/R_45_1_2.mat	f0', inplace=True)
n_folds = 10
dataset_test[['class','mat_name']] = dataset_test['processed_video_name'].str.split('/', expand=True)

dict_fold_list = {fold:list((dataset_test[dataset_test['fold']==fold]['mat_name']).unique()) for fold in range(n_folds)}
num_frames = 6
verbose = False

#%% main



dataset_in_path = '/mnt/disk2/diego.gragnaniello/Eco/ICPR/Dataset_processato/Dataset_f/' if on_cuda else '/Volumes/SD Card/Thesis/ICPR/Dataset_processato/Dataset_f/'
root_out_path = f'/home/luigi.damico/ICPR_datasets/ICPR_rawframes_numframes{num_frames}/' if on_cuda else f'/Volumes/SD Card/Thesis/ICPR/Dataset_rawframe/ICPR_numframes{num_frames}/'
classes = ['BEST', 'RDS']
phases = ['train', 'val', 'test']


# main -> fold_test -> phase -> class -> video -> img
os.mkdir(root_out_path)
for fold_test in range(n_folds):
    print('Starting fold_test: ' + str(fold_test))
    fold_out_path = root_out_path + 'foldtest_' + str(fold_test) + '/'
    os.mkdir(fold_out_path)

    for phase in phases:
        print('-- Starting for phase: ' + phase)
        phase_out_path = fold_out_path + 'rawframes_' + phase + '/'
        os.mkdir(phase_out_path) 
        if phase == 'train':
            video_list = []
            for idx in range(fold_test+1, fold_test+1 + n_folds-2):
                fold_train = idx % 10
                video_list = video_list + dict_fold_list[fold_train]
        elif phase == 'val':
            video_list = dict_fold_list[(fold_test - 1 + n_folds) % n_folds]
        elif phase == 'test':
            video_list = dict_fold_list[fold_test]
        df_annotation = pd.DataFrame(columns=['frame_name', 'n_frames', 'class'])
        df_informations = pd.DataFrame(columns=['processed_video_name', 'bimbo_name', 'classe', 'esame_name', 'paziente', 'emogas_index', 'video_name', 'frame_key', 'total_clip_frames'])
        
        for class_ in [class_ for class_ in os.listdir(dataset_in_path) if class_ in classes]:
            print('---- Starting for class: ' + class_)
            class_in_path = dataset_in_path + class_ + '/'
            class_out_path = phase_out_path + class_ + '/'
            os.mkdir(class_out_path)
            for video in [video for video in os.listdir(class_in_path) if video in video_list]:
                # print(video)
                video_in_path = class_in_path + video
                video_np, info_dict = default_mat_loader(video_in_path, num_frames=num_frames, verbose=verbose)
                df_informations = df_informations.append(info_dict, ignore_index=True)
                video_out_path = class_out_path + video[:-4] + '/'
                os.mkdir(video_out_path)
                df_annotation = df_annotation.append({'frame_name':class_+'/'+video[:-4], 'n_frames':len(video_np), 'class':1 if class_=='RDS' else 0}, ignore_index=True)
                for idx, img_np in enumerate(video_np):
                    img_out_path = video_out_path + f'img_{(idx+1):05d}.jpg'
                    cv2.imwrite(img_out_path, img_np)
        df_annotation.to_csv(fold_out_path + 'ICPR_' + phase + '_list_rawframes.txt', index=False, sep=' ', header=False)
        df_informations.to_csv(fold_out_path + 'ICPR_' + phase + '_list_informations.csv', index=False)
    

        


    
    
#%% Trial of save-loading image
# path = '/Volumes/SD Card/Thesis/ICPR/Dataset_processato/Dataset_f/RDS/R_24_1_2.mat'
# img_np = default_mat_loader(path,mode='entire_clip')[0][0]
# cv2.imwrite('prova/ciao.jpg', img_np)

    
# img_np_post = cv2.imread('prova/ciao.jpg')
# cv2.imshow('lol',img)
    
    
    
    