#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='cifar100'
PROJECT='../out/netruns'
EPOCHS=2000
BATCHSIZE=100
LEARNING_RATE=0.1
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=1
RESUME='chk000000.pth.tar'
GPU=0
ARCH='preactresnet34'
LOSS='cross'
OPT='sgd'
SCHEDULER='step'
SNAPSHOT=5
NUMCLASS=100
NUMCHANNELS=3
IMAGESIZE=32
EXP_NAME='exp_baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_aug_001'

#rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
#rm -rf $PROJECT/$EXP_NAME/
#mkdir $PROJECT
#mkdir $PROJECT/$EXP_NAME


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
--finetuning \
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

