# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, filter_images, group_images

classes = {
    0: 'background',
    1: 'building',
    2: 'road',
    3: 'pavement', #人行道
    4: 'vegetation', #植被
    5: 'baresoil',
    6: 'water'
}
# np.array([(0,0,0), (255, 0, 0), (255, 255,0), (192, 192, 0), (0,255,0), (128,128,128), (0,0,255)]


class whdldSegmentation(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        
        root = os.path.expanduser(root)
        ade_root = root
        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(ade_root, 'annotations', split)
        image_folder = os.path.join(ade_root, 'images', split)

        self.images = []
        fnames = sorted(os.listdir(image_folder))
        self.images = [
            (os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png"))
            for x in fnames
        ]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class whdldSegmentationIncremental(data.Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        masking=True,
        overlap=True,
        data_masking="current",
        ignore_test_bg=False,
        **kwargs
    ):

        full_data = whdldSegmentation(root, train)

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels) #该函数见line168，目的是移除labels中的0，即背景类。
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            self.labels = labels
            self.labels_old = labels_old

            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            #if train:
            #    masking_value = 0
            #else:
            #    masking_value = 255

            #self.inverted_order = {label: self.order.index(label) for label in self.order}
            #self.inverted_order[0] = masking_value

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            if ignore_test_bg:
                masking_value = 255
                self.inverted_order[0] = masking_value
            else:
                masking_value = 0  # Future classes will be considered as background.
            self.inverted_order[255] = 255

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if masking:
                target_transform = tv.transforms.Lambda(
                    lambda t: t.
                    apply_(lambda x: self.inverted_order[x] if x in self.labels else masking_value)
                )
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_data, idxs, transform, target_transform)
        else:
            self.dataset = full_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)
