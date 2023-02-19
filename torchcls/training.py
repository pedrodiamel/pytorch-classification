import datetime
import os
import random

import cv2
import numpy as np

# TORCH MODULE
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

# TORCHVISION MODULE
from torchvision import transforms

# LOCAL MODULES
from .configs.train_config import Config
from .datasets.factory import Dataset, FactoryDataset
from .neuralnet import NeuralNetClassifier
from .transforms import transforms as mtrans


def remove_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.isfile(file_path):
                os.remove(file_path)


def train(cfg: Config):

    print("NN Clasification {}!!!".format(datetime.datetime.now()))
    random.seed(cfg.seed)

    project_path = os.path.expanduser(cfg.project.path)
    if os.path.isdir(project_path) is not True:
        os.makedirs(project_path)

    project_name = "{}_{}_{}_{}_{}_v{}".format(
        cfg.project.name_prefix,
        cfg.trainer.arch,
        cfg.trainer.loss,
        cfg.trainer.opt,
        cfg.data.name_dataset,
        cfg.project.version,
    )

    # Check path project
    # TODO February 18, 2023: Check this function for include in pytvision
    project_pathname = os.path.expanduser(os.path.join(project_path, project_name))
    if os.path.exists(project_pathname) is not True:
        os.makedirs(project_pathname)
    else:
        response = input("Do you want to remove all files in this folder? (y/n): ")
        if response.lower() == "y":
            remove_files(project_path)

    # Create and load neural net training class
    # TODO February 18, 2023: I think that we can input the cfgs
    network = NeuralNetClassifier(
        patchproject=project_path,
        nameproject=project_name,
        no_cuda=not cfg.trainer.cuda,
        parallel=cfg.trainer.parallel,
        seed=cfg.seed,
        print_freq=cfg.checkpoint.print_freq,
        gpu=cfg.trainer.gpu,
    )

    network.create(
        arch=cfg.trainer.arch,
        num_output_channels=cfg.trainer.numclass,
        num_input_channels=cfg.trainer.numchannels,
        loss=cfg.trainer.loss,
        lr=cfg.trainer.lr,
        momentum=cfg.trainer.momentun,
        optimizer=cfg.trainer.opt,
        lrsch=cfg.trainer.scheduler,
        pretrained=cfg.trainer.finetuning,
        topk=(1,),
        size_input=cfg.trainer.image_size,
    )

    # Set cuda cudnn benchmark true
    cudnn.benchmark = True

    # Resume model
    if cfg.checkpoint.resume:
        network.resume(os.path.join(network.pathmodels, cfg.checkpoint.resume))

    print("Load model: ")
    # print(network)

    # Load dataset
    size_input = cfg.trainer.image_size

    # Load training dataset
    train_data = Dataset(
        data=FactoryDataset.factory(
            pathname=cfg.data.path,
            name=FactoryDataset.str_to_dataset[cfg.data.name_dataset],
            subset=FactoryDataset.Subsets.TRAIN,
            download=True,
        ),
        count=1000,
        num_channels=network.num_input_channels,
        transform=transforms.Compose(
            [
                mtrans.ToResize((48, 48), resize_mode="squash", padding_mode=cv2.BORDER_REPLICATE),
                mtrans.RandomScale(factor=0.1, padding_mode=cv2.BORDER_REPLICATE),
                # mtrans.ToRandomTransform(
                #     mtrans.RandomGeometricalTransform(
                #         angle=15, translation=0.1, warp=0.001, padding_mode=cv2.BORDER_REPLICATE
                #     ),
                #     prob=0.2,
                # ),
                mtrans.ToRandomTransform(mtrans.HFlip(), prob=0.5),
                # Color transformation
                # mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.05 ), prob=0.50 ),
                # mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.05 ), prob=0.50 ),
                # mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.05 ), prob=0.50 ),
                # mtrans.CLAHE(clipfactor=2.0, tileGridSize=(4, 4)),
                mtrans.ToGrayscale(),
                mtrans.ToResize(
                    (size_input + 2, size_input + 2), resize_mode="squash", padding_mode=cv2.BORDER_REPLICATE
                ),
                mtrans.RandomCrop((size_input, size_input), limit=2, padding_mode=cv2.BORDER_REPLICATE),
                # mtrans.ToEqNormalization([size_input, size_input]),
                mtrans.ToTensor(),
                mtrans.ToNormalization(),
            ]
        ),
    )

    num_train = len(train_data)
    if cfg.data.auto_balance:
        _, counts = np.unique(train_data.labels, return_counts=True)
        weights = 1 / (counts / counts.sum())
        samples_weights = np.array([weights[x] for x in train_data.labels])
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    else:
        sampler = SubsetRandomSampler(np.random.permutation(num_train))

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=cfg.data.workers,
        pin_memory=network.cuda,
        drop_last=True,
    )

    # Load validation dataset
    val_data = Dataset(
        data=FactoryDataset.factory(
            pathname=cfg.data.path,
            name=FactoryDataset.str_to_dataset[cfg.data.name_dataset],
            subset=FactoryDataset.Subsets.VAL,
            download=True,
        ),
        count=100,
        num_channels=network.num_input_channels,
        transform=transforms.Compose(
            [
                mtrans.ToResize((size_input, size_input), resize_mode="squash", padding_mode=cv2.BORDER_REPLICATE),
                # Color transformation
                # mtrans.ToRandomTransform( mtrans.RandomBrightness( factor=0.05 ), prob=0.50 ),
                # mtrans.ToRandomTransform( mtrans.RandomContrast( factor=0.05 ), prob=0.50 ),
                # mtrans.ToRandomTransform( mtrans.RandomGamma( factor=0.05 ), prob=0.50 ),
                # mtrans.CLAHE(clipfactor=2.0, tileGridSize=(4, 4)),
                mtrans.ToGrayscale(),
                # mtrans.ToEqNormalization([size_input, size_input]),
                mtrans.ToTensor(),
                mtrans.ToNormalization(),
            ]
        ),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.workers,
        pin_memory=network.cuda,
        drop_last=False,
    )

    print("Load datset")
    print("Train: ", len(train_data))
    print("Val: ", len(val_data))

    # training neural net
    network.fit(train_loader, val_loader, cfg.data.epochs, cfg.checkpoint.snapshot)

    print("Optimization Finished!")
    print("DONE!!!")
