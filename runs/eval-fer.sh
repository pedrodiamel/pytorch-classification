#!/bin/bash


PATHDATASET='~/.datasets/'
PROJECT='../out/baselinenetruns'
PROJECTNAME='baseline_dexpression_cross_adam_ferp_weights_000'
PATHNAMEOUT='../out'
FILENAME='result_fer.txt'
PATHMODEL='models'
NAMEMODEL='model_best.pth.tar' #'model_best.pth.tar' #'chk000565.pth.tar'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL  

python ../eval_fer.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--pathnameout=$PATHNAMEOUT \
--filename=$FILENAME \
--model=$MODEL \