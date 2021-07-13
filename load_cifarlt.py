from data.dataloader import *
from utils import source_import, get_value

splits = ['train', 'test', 'val']

data_root ='/home/vision/jihun/fb_decouple/dataset/cifar-100',
from data import dataloader
from data.CIFAR100_LT.imbalance_cifar import IMBALANCECIFAR100

data = {x: dataloader.load_data(data_root=data_root,
                                    dataset='cifar100_lt', phase=x,
                                    batch_size=128,
                                    sampler_dic=None,
                                    num_workers=0)
            for x in splits}