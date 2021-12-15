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
NUM_ROWS = 224
NUM_FRAMES = 2

on_cuda = torch.cuda.is_available()


def default_mat_loader(path, num_rows=NUM_ROWS, return_value=False, mode='fixed_number_of_frames', get_information=False):

    matdata = loadmat(path)
    valore = matdata['valore']
        
    frame_keys = []
    if mode == 'fixed_number_of_frames':
        count_frames = 0
        data = []
        for k in matdata.keys():
            if k.startswith('f') and len(k) < 3 and count_frames < NUM_FRAMES:
                # print(k)
                count_frames += 1
                data.append(matdata[k][:num_rows])
                frame_keys.append(k)
        data = np.array(data)
    elif mode == 'fixed_number_of_frames_1ch':
        data = [matdata[k][:num_rows] for k in matdata.keys() if k.startswith('f') and len(k) < 3]
        data = data[:NUM_FRAMES]
        data = np.array(data)
        data = np.delete(data, 0, 3)
        data = np.delete(data, 0, 3)
    elif mode == 'entire_clip':
        data = []
        for k in matdata.keys():
            if k.startswith('f') and len(k) < 3:
                data.append(matdata[k][:num_rows])
                frame_keys.append(k)
        data = np.array(data)
    elif mode == 'random_frame_from_clip' or mode == 'random_frame_from_clip_old':
        f = choice([k for k in matdata.keys() if k.startswith('f') and len(k) < 3])
        data = matdata[f][:num_rows]

    
    if get_information:
        if len(frame_keys) > 0:
            processed_video_path = path.split('/')
            information_dit = [{
                'bimbo_name': str(matdata['bimbo_name'][0]),
                'classe': str(matdata['classe'][0]),
                'esame_name': str(matdata['esame_name'][0]),
                'paziente': str(matdata['paziente'][0][0]),
                'valore': str(matdata['valore'][0][0]),
                'video_name': str(matdata['video_name'][0]),
                'processed_video_name': processed_video_path[-2] + '/' + processed_video_path[-1],
                'frame_key': k, 
                'total_clip_frames': len(data)
                } for k in frame_keys]
        else:
            processed_video_path = path.split('/')
            information_dit = {
                'bimbo_name': str(matdata['bimbo_name'][0]),
                'classe': str(matdata['classe'][0]),
                'esame_name': str(matdata['esame_name'][0]),
                'paziente': str(matdata['paziente'][0][0]),
                'valore': str(matdata['valore'][0][0]),
                'video_name': str(matdata['video_name'][0]),
                'processed_video_name': processed_video_path[-2] + '/' + processed_video_path[-1],
                'frame_key': 'None', 
                'total_clip_frames': len(data)
            }
        return data, float(valore.item()/480.), information_dit
    else:
        return data, float(valore.item()/480.)
    
    
#%%

dataset_test = pd.read_csv('/home/luigi.damico/DL_QLUS/Experiments/experiment_allfold_exp_0/evaluation_dataframe.csv' if on_cuda else'/Volumes/SD Card/Thesis/Experiments/models_training/experiment_allfold_exp_0/evaluation_dataframe.csv')
n_folds = 10
dataset_test[['class','mat_name']] = dataset_test['processed_video_name'].str.split('/', expand=True)

dict_fold_list = {fold:list((dataset_test[dataset_test['fold']==fold]['mat_name']).unique()) for fold in range(n_folds)}


#%% main



dataset_in_path = '/mnt/disk2/diego.gragnaniello/Eco/ICPR/Dataset_processato/Dataset_f/' if on_cuda else '/Volumes/SD Card/Thesis/ICPR/Dataset_processato/Dataset_f/'
main_out_path = '/home/luigi.damico/ICPR/' if on_cuda else '/Volumes/SD Card/Thesis/ICPR/Dataset_rawframe/ICPR/'
classes = ['BEST', 'RDS']
phases = ['train', 'val', 'test']


# main -> fold_test -> phase -> class -> video -> img
for fold_test in range(n_folds):
    print('Starting fold_test: ' + str(fold_test))
    fold_out_path = main_out_path + 'foldtest_' + str(fold_test) + '/'
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
        
        for class_ in [class_ for class_ in os.listdir(dataset_in_path) if class_ in classes]:
            print('---- Starting for class: ' + class_)
            class_in_path = dataset_in_path + class_ + '/'
            class_out_path = phase_out_path + class_ + '/'
            os.mkdir(class_out_path)
            for video in [video for video in os.listdir(class_in_path) if video in video_list]:
                # print(video)
                video_in_path = class_in_path + video
                video_np = default_mat_loader(video_in_path,mode='fixed_number_of_frames')[0]
                video_out_path = class_out_path + video[:-4] + '/'
                os.mkdir(video_out_path)
                df_annotation = df_annotation.append({'frame_name':class_+'/'+video[:-4], 'n_frames':len(video_np), 'class':1 if class_=='RDS' else 0}, ignore_index=True)
                for idx, img_np in enumerate(video_np):
                    img_out_path = video_out_path + f'img_{(idx+1):05d}.jpg'
                    cv2.imwrite(img_out_path, img_np)
        df_annotation.to_csv(fold_out_path + 'ICPR_' + phase + '_list_rawframes.txt', index=False, sep=' ', header=False)
        
    

        


    
    
#%% Trial of save-loading image
# path = '/Volumes/SD Card/Thesis/ICPR/Dataset_processato/Dataset_f/RDS/R_24_1_2.mat'
# img_np = default_mat_loader(path,mode='entire_clip')[0][0]
# cv2.imwrite('prova/ciao.jpg', img_np)

    
# img_np_post = cv2.imread('prova/ciao.jpg')
# cv2.imshow('lol',img)
    
    
    
    