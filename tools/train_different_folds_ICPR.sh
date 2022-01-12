#!/usr/bin/env bash

### To edit ###
NUM_EPOCHS=18
NUM_FOLDS=10
NUM_FRAMES_PER_VIDEO=2
EXPERIMENT_NAME=27
KEEP_ONLY_BEST=0
VIDEOS_PER_GPU_TRAIN=2
###############

DATASET_DIR_IN=/home/luigi.damico/ICPR_datasets/ICPR_rawframes_allFrames/
WORK_DIR=/home/luigi.damico/Video-Swin-Transformer/work_dirs/experiment_${EXPERIMENT_NAME}/
TRAINING_DIR=${WORK_DIR}training/
PREDICTION_DIR_OUT=${WORK_DIR}predictions/
RESULTS_DIR_OUT=${WORK_DIR}results/
PHASES='val test'


echo ---- Starting training phase
for FOLD_TEST in $(seq 0 $((NUM_FOLDS-1)))
do
  python tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_ICPR_1k_rawframe.py  \
  --cfg-options total_epochs=${NUM_EPOCHS} \
  work_dir=${TRAINING_DIR}testfold_${FOLD_TEST} \
  data.train.data_prefix=${DATASET_DIR_IN}foldtest_${FOLD_TEST}/rawframes_train \
  data.train.ann_file=${DATASET_DIR_IN}foldtest_${FOLD_TEST}/ICPR_train_list_rawframes.txt \
  data.videos_per_gpu=${VIDEOS_PER_GPU_TRAIN}
  #data.train.pipeline.0.clip_len=${NUM_FRAMES_PER_VIDEO}
done


echo ---- Starting evaluation phase
for PHASE in $PHASES
do
  for FOLD_TEST in $(seq 0 $((NUM_FOLDS-1)))
  do
    for EPOCH in $(seq 1 $NUM_EPOCHS)
    do
      echo Evaluating... phase ${PHASE}, fold ${FOLD_TEST}, epoch ${EPOCH}
      python tools/test.py configs/recognition/swin/swin_tiny_patch244_window877_ICPR_1k_rawframe.py \
      ${TRAINING_DIR}testfold_${FOLD_TEST}/epoch_${EPOCH}.pth \
      --eval top_k_accuracy \
      --out ${PREDICTION_DIR_OUT}${PHASE}/testfold_${FOLD_TEST}/epoch_${EPOCH}.pkl \
      --cfg-options data.test.data_prefix=${DATASET_DIR_IN}foldtest_${FOLD_TEST}/rawframes_${PHASE} \
      data.test.ann_file=${DATASET_DIR_IN}foldtest_${FOLD_TEST}/ICPR_${PHASE}_list_rawframes.txt
      #data.val.pipeline.0.clip_len=${NUM_FRAMES_PER_VIDEO}
      #data.test.pipeline.0.clip_len=${NUM_FRAMES_PER_VIDEO}
    done
  done
done


echo ---- Generating results csv
python results_analysis/generate_results_csv.py \
${WORK_DIR} \
${DATASET_DIR_IN} \
${NUM_EPOCHS} \
${NUM_FOLDS} \
${RESULTS_DIR_OUT} \
${KEEP_ONLY_BEST}
