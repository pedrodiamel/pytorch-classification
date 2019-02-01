#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='bu3dfe'
PROJECT='../out/netruns'
EPOCHS=1000
BATCHSIZE=128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=20
RESUME='chk000000.pth.tar'
GPU=0
ARCH='preactresnet18'
LOSS='cross'
OPT='adam'
SCHEDULER='step'
SNAPSHOT=50
NUMCLASS=7
NUMCHANNELS=3
IMAGESIZE=32
KFOLD=0
NACTOR=10
EXP_NAME='ferbase_'$ARCH'_'$LOSS'_'$OPT'_real_'$NAMEDATASET'_fold'$KFOLD'_000'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT
mkdir $PROJECT/$EXP_NAME

## execute
python ../train.py \
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

