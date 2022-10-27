import os
import argparse
import datetime
import logging
import numpy as np
from collections import OrderedDict, defaultdict

import torch
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import  WeightedRandomSampler
import torchvision.models as models

from model import *
from data_loader import CreativeDataset
from item_info import item_info


def main(args):

    # build model
    base_model = models.resnet18(pretrained=False, num_classes=args.feature_dim)
    model = VAM(base_model, args.feature_dim)

    # load pretrained model
    if args.resume:
        try:
            state_dict = torch.load(args.resume)
            state_dict = {k:(v[:args.feature_dim] if 'fc' in k else v) for (k,v) in state_dict.items()}
            model.features.load_state_dict(state_dict)
            print ('Successfully load the pretrained weights.')
        except IOError:
            print ("Unable to open file {}".format(args.resume))


    # init distributing env
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.local_rank == 0:
        print ("args: ", args)
        print ("cuda: {}".format(torch.cuda.is_available()))
        print("word_size: {}".format(dist.get_world_size()))
        print("master rank: {}".format(dist.get_rank()))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank)
    optimizer =  optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    #data argumentation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    # generate training samples
    train_list = gen_train_list(args)
    trainset = CreativeDataset(data_list=train_list,
                               transform=train_transform,
                               args=args)

    if args.datasampler == 'weigtedsampler':
        train_sample_weight = trainset.get_train_weight()
        train_sampler = WeightedRandomSampler(train_sample_weight,
                                              len(train_sample_weight)//dist.get_world_size(),
                                              replacement=False)
    elif args.datasampler == 'distributedsampler':
        train_sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank())
    else:
        raise NotImplementedError(args.datasampler+" has not been implemented!")

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers)
    train_iters = len(trainset) // args.batch_size // dist.get_world_size() + 1

    # begin to train
    loss = 0.
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        loss = train(train_loader, model, optimizer, epoch, train_iters, args)
        if str(dist.get_rank())=='0':
            torch.save(model.module.state_dict(), os.path.join(args.save_folder, 'vam_model_epoch_'+str(epoch)+'.pkl'))

    #reduced_loss = torch.Tensor([loss*1.0 / dist.get_world_size()]).cuda()
    #dist.all_reduce(reduced_loss)
    print('Training completed.')

def gen_train_list(args):
    train_items = defaultdict(item_info)
    with open (args.train_list_file) as f:
        for line in f:
            line = line.strip('\n').split('\t')
            train_items[line[0]].add_creative(line[1], int(line[3]), int(line[4]))
    if args.local_rank  == 0:
        print("{} items in training list.".format(len(train_items.keys())))

    filtered_train_list = []
    for item in train_items:
        train_items[item].filter_creative(args.pv_thresh)
        if train_items[item].creative_num >= 2:
            filtered_train_list.append(train_items[item])
    if args.local_rank  == 0:
        print("{} items in flitered training list.".format(len(filtered_train_list)))

    return  filtered_train_list

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, optimizer, epoch, train_iters, args):
    model.train()
    for i, data in enumerate(train_loader):
        images = data['image'].cuda().float()
        images = images.view(-1, 3, args.crop_size, args.crop_size)
        rank_label = data['rank_label'].cuda().float()
        point_label = data['point_label'].cuda().float()

        outputs = model(images)
        outputs = outputs.view(-1, args.list_len)
        optimizer.zero_grad()
        #calculate loss
        criterion = nn.LogSoftmax(dim=1)
        rank_label = rank_label.view(-1, args.list_len)
        rank_loss = - torch.sum(criterion(outputs) * rank_label.detach()) / outputs.size(0)
        if args.point_aux:
            aux_criterion = nn.MSELoss()
            point_label = point_label.view(-1, args.list_len)
            point_loss = aux_criterion(outputs, point_label)
            loss = rank_loss + 0.5*point_loss
        else:
            loss = rank_loss

        loss.backward()
        optimizer.step()

        if i % 100 ==0:
            time_stamp = datetime.datetime.now()
            if args.local_rank ==  0:
                if args.point_aux:
                    print(time_stamp.strftime('%Y.%m.%d-%H:%M:%S '), 'Epoch: %d/%d | Step: %d/%d | Training rank/point loss: %.4f/%.4f' %
                        (epoch + 1, args.epochs, i + 1, train_iters, rank_loss.item(), point_loss.item()))
                else:
                    print(time_stamp.strftime('%Y.%m.%d-%H:%M:%S '), 'Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
                        (epoch + 1, args.epochs, i + 1, train_iters, rank_loss.item()))

    return loss.item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch table IO')
    parser.add_argument('--train-list-file', default="data/train_data_list.txt", type=str, help='ODPS input table names')
    parser.add_argument('--image-folder', default="www/images", type=str, help='ODPS input table names')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for each GPU')
    parser.add_argument('--feature-dim', default="10", type=int, help='feature dimension for last fc layer. It should be <= 1000')
    parser.add_argument('--img-size', default="256", type=int, help='resized size for images')
    parser.add_argument('--crop-size', default="224", type=int, help='cropped size for images')
    parser.add_argument('--datasampler', default="weigtedsampler", type=str, help='weigtedsampler or distributedsampler')
    parser.add_argument('--pv-thresh', default="60", type=int, help='cut-off threshold for each image pv')
    parser.add_argument('--list-len', default="3", type=int, help='length of the list')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, help='Number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--alpha', default=2.0402, type=float, help='alpha for beta smooth')
    parser.add_argument('--beta', default=54.0625, type=float, help='beta for beta smooth')
    parser.add_argument('--resume', default='VAM/resnet18.pth', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--local_rank', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--point-aux', dest='point_aux', action='store_false', help='ctr regression as auxiliary info')
    parser.add_argument('--save-folder', default="weights", type=str, help='folders for saving models')

    args = parser.parse_args()
    main(args)

