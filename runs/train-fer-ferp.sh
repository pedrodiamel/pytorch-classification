#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='affectnet' #ferp, affectnet
PROJECT='../out/baselinenetruns'
EPOCHS=150
BATCHSIZE=128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=75
WORKERS=20
RESUME='chk000149.pth.tar'
GPU=0
ARCH='ferattentionstn' #preactresnet18, fmp, cvgg13, resnet18, alexnet, dexpression, ferattention, ferattentionstn
LOSS='cross'
OPT='adam'
SCHEDULER='step'
SNAPSHOT=10
NUMCLASS=8
NUMCHANNELS=3
IMAGESIZE=64
EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_cvgg13_weights_norm_000'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT
mkdir $PROJECT/$EXP_NAME

## execute
CUDA_VISIBLE_DEVICES=2,3 python ../train.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
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
--finetuning \
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

