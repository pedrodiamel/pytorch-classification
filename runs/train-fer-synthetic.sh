#!/bin/bash

# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='affectnetdark' #affectnetdark,  
PROJECT='../out/netruns'
EPOCHS=150
BATCHSIZE=128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=0
RESUME='chk000000xxx.pth.tar'
GPU=0
ARCH='preactresnet18'
LOSS='cross'
OPT='adam'
SCHEDULER='step'
SNAPSHOT=50
NUMCLASS=8
NUMCHANNELS=3
IMAGESIZE=32
KFOLD=0
NACTOR=10
EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_fold'$KFOLD'_weights_000'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT
mkdir $PROJECT/$EXP_NAME

## execute
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../train_fersynthetic.py \
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

