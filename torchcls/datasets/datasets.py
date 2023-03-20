import os
import random

import warnings
from collections import namedtuple

import numpy as np

import torch
from pytvision.datasets import utility
from pytvision.transforms.aumentation import ObjectImageAndLabelTransform, ObjectImageTransform

warnings.filterwarnings("ignore")


class Dataset(object):
    """
    Generic dataset
    """

    def __init__(self, data, num_channels=1, count=None, transform=None):
        """
        Initialization
        Args:
            @data: dataprovide class
            @num_channels:
            @tranform: tranform
        """

        if count is None:
            count = len(data)
        self.count = count
        self.data = data
        self.num_channels = num_channels
        self.transform = transform
        self.labels = data.labels
        self.classes = np.unique(self.labels)
        self.numclass = len(self.classes)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        idx = idx % len(self.data)
        image, label = self.data[idx]
        image = np.array(image)
        image = utility.to_channels(image, self.num_channels)
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform(image, label)
        if self.transform:
            obj = self.transform(obj)
        return obj.to_dict()


class ResampleDataset(object):
    """
    Resample data for generic dataset
    """

    def __init__(self, data, num_channels=1, count=200, transform=None):
        """
        Initialization
        data: dataloader class
        tranform: tranform
        """

        self.num_channels = num_channels
        self.data = data
        self.transform = transform
        self.labels = data.labels
        self.count = count

        # self.classes = np.unique(self.labels)
        self.classes, self.frecs = np.unique(self.labels, return_counts=True)
        self.numclass = len(self.classes)

        # self.weights = 1-(self.frecs/np.sum(self.frecs))
        self.weights = np.ones((self.numclass, 1))
        self.reset(self.weights)

        self.labels_index = list()
        for cl in range(self.numclass):
            indx = np.where(self.labels == cl)[0]
            self.labels_index.append(indx)

    def reset(self, weights):
        self.dist_of_classes = np.array(random.choices(self.classes, weights=weights, k=self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        idx = self.dist_of_classes[idx]
        class_index = self.labels_index[idx]
        n = len(class_index)
        idx = class_index[random.randint(0, n - 1)]

        image, label = self.data[idx]

        image = np.array(image)
        image = utility.to_channels(image, self.num_channels)
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform(image, label)
        if self.transform:
            obj = self.transform(obj)
        return obj.to_dict()
