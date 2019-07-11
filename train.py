

# STD MODULE
import os
import numpy as np
import cv2
import random

# TORCH MODULE
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn

# PYTVISION MODULE
from pytvision.transforms import transforms as mtrans
from pytvision.datasets.datasets  import Dataset
from pytvision.datasets.factory  import FactoryDataset
from pytvision import visualization as view

# LOCAL MODULE
# from torchlib.datasets.datasets  import Dataset
# from torchlib.datasets.factory  import FactoryDataset

from torchlib.neuralnet import NeuralNetClassifier
from misc import get_transforms_aug, get_transforms_det


from argparse import ArgumentParser
import datetime

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', 
                        help='path to dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='divice number (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', 
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--project', default='./runs', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--name', default='exp', type=str,
                        help='name of experiment')
    parser.add_argument('--resume', type=str, metavar='NAME',
                        help='name to latest checkpoint (default: none)')
    parser.add_argument('--arch', default='simplenet', type=str,
                        help='architecture')
    parser.add_argument('--finetuning', action='store_true', default=False,
                        help='Finetuning')
    parser.add_argument('--loss', default='cross', type=str,
                        help='loss function')
    parser.add_argument('--opt', default='adam', type=str,
                        help='optimize function')
    parser.add_argument('--scheduler', default='fixed', type=str,
                        help='scheduler function for learning rate')
    parser.add_argument('--snapshot', '-sh', default=10, type=int, metavar='N',
                        help='snapshot (default: 10)')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Parallel')
    parser.add_argument('--num-classes', '-c', default=10, type=int, metavar='N',
                        help='num classes (default: 10)')
    parser.add_argument('--image-size', default=256, type=int, metavar='N',
                        help='image size (default: 256)')
    parser.add_argument('--name-dataset', default='mnist', type=str,
                        help='name dataset')
    parser.add_argument('--channels', default=1, type=int, metavar='N',
                        help='input channel (default: 1)')

    return parser


def main():
    
    # parameters
    parser = arg_parser();
    args = parser.parse_args();
    random.seed(0)
    
    print('Baseline clasification {}!!!'.format(datetime.datetime.now()))
    print('\nArgs:')
    [ print('\t* {}: {}'.format(k,v) ) for k,v in vars(args).items() ]
    print('')
    
    network = NeuralNetClassifier(
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        parallel=args.parallel,
        seed=args.seed,
        print_freq=args.print_freq,
        gpu=args.gpu
        )

    network.create( 
        arch=args.arch, 
        num_output_channels=args.num_classes, 
        num_input_channels=args.channels, 
        loss=args.loss, 
        lr=args.lr, 
        momentum=args.momentum,
        optimizer=args.opt,
        lrsch=args.scheduler,
        pretrained=args.finetuning,
        topk=(1, ),
        size_input=args.image_size,
        )
    
    cudnn.benchmark = True

    # resume model
    if args.resume:
        network.resume( os.path.join(network.pathmodels, args.resume ) )

    # print neural net class
    print('Load model: ')
    print(network)

    kfold=0 #args.kfold
    nactores=10 #args.nactor
    idenselect = np.arange(nactores) + kfold*nactores

    
    # datasets
    # training dataset
    train_data = Dataset(
        data=FactoryDataset.factory(
            pathname=args.data, 
            name=args.name_dataset, 
            subset=FactoryDataset.training, 
            #idenselect=idenselect,
            download=True ),
        #count=100000,
        num_channels=network.num_input_channels,
        transform=get_transforms_aug( network.size_input ), #get_transforms_aug
        )
    
    
    labels, counts = np.unique(train_data.labels, return_counts=True)
    weights = 1/(counts/counts.sum())        
    samples_weights = np.array([ weights[ x ]  for x in train_data.labels ])    

    labels, counts = np.unique(train_data.labels, return_counts=True)
    weights = 1/(counts/counts.sum())        
    samples_weights = np.array([ weights[ x ]  for x in train_data.labels ])   
    
    num_train = len(train_data)
    sampler = WeightedRandomSampler( weights=samples_weights, num_samples=len(samples_weights), replacement=True )
    #sampler = SubsetRandomSampler(np.random.permutation( num_train ) ) 
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
        sampler=sampler, num_workers=args.workers, pin_memory=network.cuda, drop_last=True)
    
    # validate dataset
    val_data = Dataset(
        data=FactoryDataset.factory(
            pathname=args.data, 
            name=args.name_dataset, 
            subset=FactoryDataset.validation, 
            #idenselect=idenselect,
            download=True ),
        #count=10000,
        num_channels=network.num_input_channels,
        transform=get_transforms_det( network.size_input ),
        )
    
    num_val = len(val_data)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.workers, pin_memory=network.cuda, drop_last=False)
       
        
    print('Load datset')
    print('Train: ', len(train_data))
    print('Val: ', len(val_data))
    
    # training neural net
    network.fit( train_loader, val_loader, args.epochs, args.snapshot )
    
    print("Optimization Finished!")
    print("DONE!!!")



if __name__ == '__main__':
    main()
