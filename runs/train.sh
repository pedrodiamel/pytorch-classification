#!/bin/bash

set -e # abort if any command fails


DATASET='affectnetdark'
MODELS=('inception_v3' 'simplenet' 'alexnet' 'vgg11' 'resnet18'  'densenet121' 'cvgg13' 'dexpression' 'fmp' 'preactresnet18')
IMAGESIZE=( 229 64 227 224 224 224 64 64 48 32 )

for i in {0..9}
do 
    echo ${MODELS[$i]} ${IMAGESIZE[$i]}x${IMAGESIZE[$i]} 
    bash train-fer-synthetic-gen.sh $DATASET ${MODELS[$i]} ${IMAGESIZE[$i]}
    
done
