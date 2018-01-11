#!/bin/python3

import argparse
import logging
import matplotlib.pyplot as plt
import torch
import torchvision

from stylizer import NeuralStylizer
CONTENT_LAYERS = ['conv_4_2']
STYLE_LAYERS = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')

def main():
    parser = argparse.ArgumentParser(usage='style transfer using neural networks.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('content_image',
                        type=str,
                        action='store',
                        help='content image')
    parser.add_argument('style_image',
                        type=str,
                        help='style image')
    parser.add_argument('-o',
                        type=str,
                        default='result.jpg',
                        action='store',
                        dest='output_file',
                        help='name of output file')
    parser.add_argument('--pooling',
                        choices=['max', 'avg'],
                        default='avg',
                        help='type of pooling layer to be used')
    parser.add_argument('--init',
                        choices=['content', 'random'],
                        default='random',
                        help='select between using the content image or a random image as the initial input')
    parser.add_argument('--backend',
                        choices=['cpu', 'cuda'],
                        default='cpu',
                        help='select from available backends')
    parser.add_argument('--target-shape',
                        type=int,
                        nargs=2,
                        help='w x h of initial input image fed to neural net')
    parser.add_argument('--output-shape',
                        type=int,
                        nargs=2,
                        help='w x h of final image, defaults to 224x224 for cpu, 512x512 for cuda if not set')
    parser.add_argument('--iterations',
                        type=int,
                        default=300,
                        help='number of iterations to run')
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='the weight of the content image in the final image')
    parser.add_argument('--beta',
                        type=float,
                        default=1000.0,
                        help='the weight of the style image in the final image')
    parser.add_argument('--tv-weight',
                        type=float,
                        default=1e-2,
                        help='scale of total variation error to minimize noise')
    args = parser.parse_args()

    term = logging.StreamHandler()
    term.setLevel(logging.INFO)
    term.setFormatter(formatter)
    logger.addHandler(term)

    enable_cuda = torch.cuda.is_available() if args.backend == 'cuda' else False
    if args.target_shape is None:
        args.target_shape = (512, 512) if enable_cuda else (224, 224)
        logger.info('target shape not set by user, set to {} x {}'.format(*args.target_shape))
    logger.info('fetching vgg19 model')
    cnn = torchvision.models.vgg19(pretrained=True).features
    if enable_cuda:
         cnn.cuda()
    logger.info('initializing model')
    stylize = NeuralStylizer(cnn, args.content_image, args.style_image,  args.target_shape, args.output_shape,
                             args.enable_cuda, CONTENT_LAYERS, STYLE_LAYERS, args.pooling,
                             args.alpha, args.beta, args.tv_weight)
    logger.info('optimizing..')
    stylize(args.iterations, args.output_shape, args.init, args.output_file)
    logger.info('complete!')
    logger.info('saving output to {}'.format(args.output_file))
main()
