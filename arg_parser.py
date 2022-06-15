from __future__ import print_function

import argparse

import distutils.util

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-j','--num-workers', default=16, type=int, help='num of fetching threads')

    #important settings:
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--optimizer', default='SGD', help='optimizer(SGD|Adam|AMSGrad)')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--lr-scheduler', default='cosine', help='learning rate scheduler(multistep|cosine)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('-b','--batch-size', default=128, type=int, help='batch size')

    parser.add_argument('--start_epoch', default=0, type=int, help='the epoch from which to resume training')
    parser.add_argument('--epochs', default=20, type=int, help='training epochs')

    parser.add_argument('-a','--arch', default='vgg11', help='architecture')
    parser.add_argument('--dataset', default='cifar10', help='dataset(cifar10|cifar100|svhn|stl10|mnist)')

    parser.add_argument('--init', default='kaiming_1', help='initialization method (casnet|xavier|kaiming_1||kaiming_2)')

    parser.add_argument('--method', default=3, type=int, help='method/model type')
    parser.add_argument('--batchnorm', default=True, type=distutils.util.strtobool, help='turns on or off batch normalization')

    # for deconv
    parser.add_argument('--deconv', default=False, type=distutils.util.strtobool, help='use deconv')
    parser.add_argument('--delinear', default=True, type=distutils.util.strtobool, help='use decorrelated linear')

    parser.add_argument('--block-fc','--num-groups-final', default=0, type=int, help='number of groups in the fully connected layers')
    parser.add_argument('--block', '--num-groups', default=64,type=int, help='block size in deconv')
    parser.add_argument('--deconv-iter', default=5,type=int, help='number of iters in deconv')
    parser.add_argument('--eps', default=1e-5,type=float, help='for regularization')
    parser.add_argument('--bias', default=True,type=distutils.util.strtobool, help='use bias term in deconv')
    parser.add_argument('--stride', default=3, type=int, help='sampling stride in deconv')
    parser.add_argument('--freeze', default=False, type=distutils.util.strtobool, help='freeze the deconv updates')

    args = parser.parse_args()

    return args
