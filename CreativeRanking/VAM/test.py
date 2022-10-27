import os
import argparse
import datetime
import logging
import random
import numpy as np
from collections import OrderedDict, defaultdict

import torch
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models

from model import *
from data_loader import TestDataset
from item_info import item_info


def main(args):

    # build model
    base_model = models.resnet18(pretrained=False, num_classes=args.feature_dim)
    model = TestVAM(base_model, args.feature_dim)
    # load pretrained model
    try:
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)
        print ('Successfully load the pretrained weights.')
    except IOError:
        print ("Unable to open file {}".format(args.resume))

    model = model.cuda()

    #data argumentation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    # generate test samples
    test_list = open(args.test_list_file).read().split('\n')[:-1]
    testset = TestDataset(data_list=test_list, transform=test_transform, args=args)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers)
    test_iters = len(testset) // args.batch_size  + 1

    # begin to inference
    print ('Begin to extract image representation.')
    inference(test_loader, model, test_iters, args)
    print ('Feature extraction is done. Calculating SCTR...')
    calc_test_sctr(args)

    print('inference is completed.')


def inference(test_loader, model, test_iters, args):
    model.eval()
    fid = open(args.save_file, 'w')
    for i, data in enumerate(test_loader):
        images = data['image_content'].cuda().float()
        images = images.view(-1, 3, args.crop_size, args.crop_size)
        feats, scores = model(images)
        feats = feats.view(-1, args.feature_dim)
        scores = scores.view(-1)
        for num in range(feats.shape[0]):
            fid.write(data['item_id'][num]+' '+data['image'][num]+' '+data['ds'][num]+' '+data['pv'][num]+' '+data['clk'][num]+' ')
            fid.write(','.join(feats[num].detach().cpu().numpy().astype('str')))
            fid.write('\n')

        if i % 100 == 0:
            time_stamp = datetime.datetime.now()
            print(time_stamp.strftime('%Y.%m.%d-%H:%M:%S '), 'Iterations: %d/%d' % (i, test_iters))
    fid.close()

class best_creative(object):
    def __init__(self):
        self.pv = 0
        self.clk = 0
        self.score = 0

def calc_test_sctr(args):
    dic = defaultdict(best_creative)
    with open(args.save_file) as f:
        for line in f:
            item_id, image, ds, pv, clk, feat = line.strip('\n').split(' ')
            score = np.mean([float(x) for x in feat.split(',')])
            if score > dic[item_id].score:
            #if random.random()<0.5:
                dic[item_id].pv = int(pv)
                dic[item_id].clk = int(clk)
                dic[item_id].score = score

    total_pv = 0
    total_clk = 0
    for node in dic:
        total_pv += dic[node].pv
        total_clk += dic[node].clk
    print total_pv, total_clk, total_clk*1.0/total_pv



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch table IO')
    parser.add_argument('--test-list-file', default="test_data_list.txt", type=str, help='ODPS input table names')
    parser.add_argument('--image-folder', default="www/images", type=str, help='ODPS input table names')
    parser.add_argument('--batch-size', default=4, type=int, help='Batch size for each GPU')
    parser.add_argument('--feature-dim', default="10", type=int, help='feature dimension for last fc layer. It should be <= 1000')
    parser.add_argument('--img-size', default="256", type=int, help='resized size for images')
    parser.add_argument('--crop-size', default="224", type=int, help='cropped size for images')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='weights/vam_model_epoch_30.pkl', type=str, help='latest checkpoint for testing')
    parser.add_argument('--local_rank', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--save-file', default="extracted_feat/extract_feat_and_score.txt", type=str, help='file for saving features/scores')

    args = parser.parse_args()
    main(args)

