# -*- coding: utf-8

import os
import random
import math
import numpy as np
from PIL import Image
import cv2
from scipy.misc import logsumexp

import torch
from torch.utils import data


class CreativeDataset(data.Dataset):
    """ training/validation dataset
        Args:
            data_list: train/val list
            transform: data transform
            output_size: return the images to the given output_size
    """
    def __init__(self, data_list, transform, args):
        self.item_list = data_list
        self.transform = transform
        self.output_size = args.crop_size
        self.list_len = args.list_len
        self.image_folder = args.image_folder
        self.alpha = args.alpha
        self.beta = args.beta

    def __len__(self):
        return len(self.item_list)

    def get_train_weight(self):
        self.train_weight = [math.log(float(x.item_pv)) for x in self.item_list]
        return self.train_weight

    def read_img(self, img_name):
        image_content = None
        success = False
        try:
            image_content = Image.open(os.path.join(self.image_folder, img_name)).convert('RGB')
            image_content = self.transform(image_content)
            success = True

        except Exception as e:
            success = False
            print(e, 'Read {} failed.'.format(img_name))

        return image_content, success

    def __getitem__(self, idx):
        data = self.item_list[idx]
        image = np.zeros((self.list_len, 3, self.output_size, self.output_size))
        rank_label = np.zeros(self.list_len)
        point_label = np.zeros(self.list_len)

        list_len = min(len(data.creatives.keys()), self.list_len)
        list_img_name = random.sample(data.creatives.keys(), list_len)
        for idx, img in enumerate(list_img_name):
            image_content, success = self.read_img(img)
            if success:
                smooth_ctr = (data.creatives[img].clk + self.alpha) / (data.creatives[img].pv + self.alpha + self.beta)
                rank_label[idx] = smooth_ctr*1000.
                point_label[idx] = smooth_ctr*100.
                image[idx] = image_content
        rank_label = np.exp(rank_label - logsumexp(rank_label))

        sample = {'image': image,
                  'rank_label': rank_label,
                  'point_label': point_label}

        return sample


class TestDataset(data.Dataset):
    """test dataset
    Args:
        table: prediction list
        transform: data transform
        output_size: return the images to the given output_size
        rank: GPUID(rankid)
        count: total number of GPUs
    """
    def __init__(self, data_list, transform, args):
        self.data_list = data_list
        self.transform = transform
        self.output_size = args.crop_size
        self.image_folder = args.image_folder

    def __len__(self):
        return len(self.data_list)

    def read_img(self, img_name):
        image_content = None
        #print (os.path.join(self.image_folder, img_name))
        try:
            image_content = Image.open(os.path.join(self.image_folder, img_name)).convert('RGB')
            image_content = self.transform(image_content)

        except Exception as e:
            image_content = np.zeros((3, self.output_size, self.output_size))
            print(e, 'Read {} failed.'.format(img_name))

        return image_content

    def __getitem__(self, idx):
        data = self.data_list[idx].split('\t')
        item_id, image, ds, pv, clk = data[0], data[1], data[2], data[3], data[4]
        image_content = np.zeros((3, self.output_size, self.output_size))
        image_content = self.read_img(image)

        sample = {'image': image,
                  'image_content': image_content,
                  'item_id': item_id,
                  'ds': ds,
                  'pv': pv,
                  'clk': clk}

        return sample

        #item_id = str(self.data[idx][0], encoding='utf-8')


