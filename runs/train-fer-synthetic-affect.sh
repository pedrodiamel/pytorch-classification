#!/bin/bash

# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='affectnetdark' #affectnetdark,  ckdark
PROJECT='../out/netruns'
EPOCHS=150
BATCHSIZE=128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=10
RESUME='chk000004.pth.tar'
GPU=0
ARCH='resnet18' #resnet18, preactresnet18
LOSS='cross'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=50
NUMCLASS=8
NUMCHANNELS=3
IMAGESIZE=224
KFOLD=0
NACTOR=10
EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_fold'$KFOLD'_weights_000'
# EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_weights_000'
# EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_synthetic_black_weights_000'

# rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
# rm -rf $PROJECT/$EXP_NAME/
# mkdir $PROJECT
# mkdir $PROJECT/$EXP_NAME

## execute
CUDA_VISIBLE_DEVICES=0,1 python ../train_fer_synthetic.py \
$DATA \
--databack=$DATABACK \
--name-dataset=$NAMEDATASET \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--kfold=$KFOLD \
--nactor=$NACTOR \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--print-freq=$PRINT_FREQ \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--num-classes=$NUMCLASS \
--channels=$NUMCHANNELS \
--image-size=$IMAGESIZE \
--parallel \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \
