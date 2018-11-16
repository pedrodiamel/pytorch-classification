#!/bin/bash

# parameters
DATA='~/.datasets/coco'
NAMEDATASET='coco'
PROJECT='../out/netruns'
EPOCHS=1000
BATCHSIZE=128
LEARNING_RATE=0.0001
MOMENTUM=0.9
PRINT_FREQ=100
WORKERS=40
RESUME='chk000000xxx.pth.tar'
GPU=0
ARCH='preactresnet18'
LOSS='cross'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=5
NUMCLASS=7
NUMCHANNELS=3
IMAGESIZE=48
EXP_NAME='baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_002'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT
mkdir $PROJECT/$EXP_NAME

## execute
python ../train_fer_synthetic.py \
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

