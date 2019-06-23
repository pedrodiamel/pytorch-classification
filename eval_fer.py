

# STD MODULE
import os
import numpy as np
import cv2
import random
import pandas as pd

# TORCH MODULE
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn

# PYTVISION MODULE
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view
from pytvision.datasets.datasets  import Dataset
from pytvision.datasets.factory  import FactoryDataset

# LOCAL MODULE
# from torchlib.datasets.datasets  import Dataset
# from torchlib.datasets.factory  import FactoryDataset

from torchlib.neuralnet import NeuralNetClassifier
from misc import get_transforms_aug, get_transforms_det


# METRICS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics


from argparse import ArgumentParser
import datetime


def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('--project',     metavar='DIR',  help='path to projects')
    parser.add_argument('--projectname', metavar='DIR',  help='name projects')
    parser.add_argument('--pathdataset', metavar='DIR',  help='path to dataset')
    parser.add_argument('--pathnameout', metavar='DIR',  help='path to out dataset')
    parser.add_argument('--filename',    metavar='S',    help='name of the file output')
    parser.add_argument('--model',       metavar='S',    help='filename model')  
    return parser


def main():
    
    parser = arg_parser();
    args = parser.parse_args();

    # Configuration
    project         = args.project
    projectname     = args.projectname 
    pathnamedataset = args.pathdataset 
    pathnamemodel   = args.model
    pathproject     = os.path.join( project, projectname )
    pathnameout     = args.pathnameout
    filename        = args.filename
    
    
    pathname = os.path.expanduser( '~/.datasets' )
    
    num_workers=0
    no_cuda=False
    parallel=False
    gpu=0
    seed=1

    
    # experiments
    experiments = [ 
        { 'name': 'ferp',      'subset': FactoryDataset.test,        'real': True },
        { 'name': 'affectnet', 'subset': FactoryDataset.validation,  'real': True },
        { 'name': 'ck',        'subset': FactoryDataset.training,    'real': True },
        { 'name': 'jaffe',     'subset': FactoryDataset.training,    'real': True },
        { 'name': 'bu3dfe',    'subset': FactoryDataset.training,    'real': True },
        ]
 
    
    # Load models
    network = NeuralNetClassifier(
        patchproject=project,
        nameproject=projectname,
        no_cuda=no_cuda,
        seed=seed,
        gpu=gpu
        )

    cudnn.benchmark = True

    # load model
    if network.load( pathnamemodel ) is not True:
        print('>>Error!!! load model')
        assert(False)  

    size_input = network.size_input
    tuplas=[]
    for  i, experiment in enumerate(experiments):

        name_dataset = experiment['name']
        subset = experiment['subset']
        breal = experiment['real']
        dataset = []

        # real dataset 
        dataset = Dataset(    
            data=FactoryDataset.factory(
                pathname=pathnamedataset, 
                name=name_dataset, 
                subset=subset, 
                #idenselect=idenselect,
                download=True 
            ),
            num_channels=3,
            transform=get_transforms_det( network.size_input ),
            )
        
        
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=num_workers )
        
        Yhat, Y = network.test( dataloader )
        df = pd.DataFrame( np.concatenate((Yhat, Y), axis=1) )
        df.to_csv( os.path.join(pathproject , '{}_{}_{}_dp.csv'.format(subset,projectname,name_dataset)), index=False, encoding='utf-8')       
        
        yhat = np.argmax( Yhat, axis=1 )
        y = Y

        acc = metrics.accuracy_score(y, yhat)
        precision = metrics.precision_score(y, yhat, average='macro')
        recall = metrics.recall_score(y, yhat, average='macro')
        f1_score = 2*precision*recall/(precision+recall)
        
        #|Name|Dataset|Cls|Acc| ...
        tupla = { 
            #'Name':projectname,  
            'Dataset': '{}({})'.format(  name_dataset,  subset ),
            'Accuracy': acc*100,
            'Precision': precision*100,
            'Recall': recall*100,
            'F1 score': f1_score*100,        
        }
        tuplas.append(tupla)
        
        
    df = pd.DataFrame(tuplas)
    df.to_csv( os.path.join( pathproject, 'experiments_cls.csv' ) , index=False, encoding='utf-8')
    print('save experiments class ...')
    
    print('DONE!!!')
        

if __name__ == '__main__':
    main()


