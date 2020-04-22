#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='ck'
PROJECT='../out/netruns'
EPOCHS=60
BATCHSIZE=128 #128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=10
RESUME='chk000000.pth.tar'
GPU=0
ARCH='resnet18' #preactresnet18, fmp, cvgg13, resnet18, alexnet, dexpression
LOSS='cross'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=50
NUMCLASS=8
NUMCHANNELS=3
IMAGESIZE=224 #preactresnet18:32, Resnet18:224, 
KFOLD=0
NACTOR=10
EXP_NAME='ferbase_'$ARCH'_'$LOSS'_'$OPT'_real_'$NAMEDATASET'_fold'$KFOLD'_000'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT
mkdir $PROJECT/$EXP_NAME

## execute
CUDA_VISIBLE_DEVICES=0,1 python ../train.py \
$DATA \
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
--name-dataset=$NAMEDATASET \
--channels=$NUMCHANNELS \
--image-size=$IMAGESIZE \
--parallel \
--finetuning \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

