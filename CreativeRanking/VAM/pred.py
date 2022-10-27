#from __future__ import print_function
import numpy as np
import argparse
import datetime
import logging
from pynvml import *


import torch
import torch.distributed as dist
import torchvision.transforms as transforms

import resnet
from data_loader import TestDataset
from model import *
import common_io

model_backbone = {
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
}

def get_device_info():
    nvmlInit()
    print ("Cuda: ", torch.cuda.is_available())
    print ("Print Device Info...")
    devicecount = nvmlDeviceGetCount()
    print ("Device Count: ", devicecount)
    for i in range(devicecount):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        print ("Device Name: ", nvmlDeviceGetName(handle))
        print ("Device Mem: ", meminfo.total)
    nvmlShutdown()

def init_dist_env():
    dist.init_process_group("nccl")
    print("Wordsize: ", dist.get_world_size())
    print("Rank: ", dist.get_rank())

def main(args):
    get_device_info()
    init_dist_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('args: ', args)

    if args.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        base_model = model_backbone[args.backbone](pretrained=False, num_classes=args.feature_dim)
    else:
        raise NotImplementedError(args.backbone+" not implemented!")

    if args.SE:
        model = TestSECARNet(base_model, args.feature_dim).to(device)
    elif args.pool:
        model = TestSimCARNet(base_model, args.feature_dim).to(device)
    else:
        model = TestCARNet(base_model, args.feature_dim).to(device)
    state_dict = torch.load('/data/volume1/best_model.pkl')
    model.load_state_dict(state_dict)
    print ('Successfully load the pretrained model.')
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.img_height and args.img_width:
        if args.crop_height and args.crop_width:
            test_transform = transforms.Compose([
                transforms.Resize((args.img_height, args.img_width)),
                transforms.CenterCrop((args.crop_height, args.crop_width)),
                transforms.ToTensor(),
                normalize])
        else:
            test_transform = transforms.Compose([
                tesatransforms.Resize((args.img_height, args.img_width)),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                normalize])
    else:
        if args.crop_height and args.crop_width:
            test_transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop((args.crop_height, args.crop_width)),
                transforms.ToTensor(),
                normalize])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                normalize])

    if args.crop_height and args.crop_width:
        output_size = (args.crop_height, args.crop_width)
    else:
        output_size = args.crop_size
    testset = TestDataset(args.tables, test_transform,
                          output_size, dist.get_rank(), dist.get_world_size(), args.hwratio)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)
    writer = common_io.table.TableWriter(args.outputs, slice_id=dist.get_rank())
    with torch.no_grad():
        batch_index = 0
        for data in test_loader:
            batch_index += 1
            if batch_index % 10 == 0:
                time_stamp = datetime.datetime.now()
                print (time_stamp.strftime('%Y.%m.%d-%H:%M:%S    ')+str(batch_index)+' batches have been predicted!')
            #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+'   begin to read data...')
            image = data['image'].cuda()
            success = data['success'].view(-1)
            output = model(image)
            scores = output.cpu().detach().numpy()
            for idx in range(output.size(0)):
                if success[idx] == True:
                    writer.write([(data['item_id'][idx], data['img_name'][idx], scores[idx])], (0, 1, 2))
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch table IO')
    parser.add_argument('--tables', default="", type=str, help='ODPS input table names')
    parser.add_argument('--outputs', default="", type=str, help='ODPS output table names')
    parser.add_argument('--batch-size', default=800, type=int, help='Batch size for each GPU')
    parser.add_argument('--workers', default=24, type=int, help='Number of data loading workers (default: 10)')
    parser.add_argument('--backbone', default="resnet18", type=str, help='backbone')
    parser.add_argument('--img-size', default="256", type=int, help='resized size for images')
    parser.add_argument('--crop-size', default="250", type=int, help='cropped size for images')
    parser.add_argument('--img-height', default=None, type=int, help='resized height for images')
    parser.add_argument('--img-width', default=None, type=int, help='resized width for images')
    parser.add_argument('--crop-height', default=None, type=int, help='cropped height for images')
    parser.add_argument('--crop-width', default=None, type=int, help='cropped width for images')
    parser.add_argument('--feature-dim', default="10", type=int, help='feature dimension for last fc layer')
    parser.add_argument('--hwratio', dest='hwratio', action='store_true', help='0.8 < (height*1.0/width) < 1.2')
    parser.add_argument('--pool', dest='pool', action='store_true', help='pooling for feature aggregration')
    parser.add_argument('--SE', dest='SE', action='store_true', help='SE attention')

    args = parser.parse_args()
    main(args)

