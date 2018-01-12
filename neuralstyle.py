#!/bin/python3
import sys
import argparse
import logging
import torch
import torchvision
import math

from stylizer import NeuralStylizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')

def main():
    parser = argparse.ArgumentParser(usage='style transfer using neural networks.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content-image',
                        type=str,
                        action='store',
                        required=True,
                        help='content image')
    parser.add_argument('--style-images',
                        type=str,
                        nargs='+',
                        required=True,
                        help='1 or more style images')
    parser.add_argument('--style-weights',
                        type=float,
                        nargs='+',
                        required=True,
                        help='the weights corresponding to each style image, all weights must sum up to 1')
    parser.add_argument('--content-layers',
                        type=str,
                        nargs='+',
                        default=['conv_4_2'],
                        help='layers in the network for content loss')
    parser.add_argument('--style-layers',
                        type=str,
                        nargs='+',
                        default=['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1'],
                        help='layers in the network for style loss')
    parser.add_argument('-o', '--output',
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
    parser.add_argument('--input-size',
                        type=int,
                        help='n x n of initial input image fed to neural net, defaults to 224x224 for cpu, 512x512 for cuda if not set')
    parser.add_argument('--output-size',
                        type=int,
                        nargs=2,
                        help='w x h of final image, defaults to size of the content image if not set')
    parser.add_argument('--iterations',
                        type=int,
                        default=300,
                        help='number of iterations to run')
    parser.add_argument('--alpha',
                        type=float,
                        dest='A',
                        default=1.0,
                        help='the weight of the content loss w.r.t the input image')
    parser.add_argument('--beta',
                        type=float,
                        dest='B',
                        default=1000.0,
                        help='the weight of the style loss w.r.t the input image')
    parser.add_argument('--lambda',
                        type=float,
                        dest='L',
                        default=1e-2,
                        help='the weight of total variation loss w.r.t noise in the input image')
    args = parser.parse_args()

    if args.content_image is None or args.style_images is None:
        logger.error('ERROR: content image or style image not set')
        logger.error('exiting...')
        sys.exit(0)

    if len(args.style_images) != len(args.style_weights):
        logger.error("ERROR: number of style weights don't match number of style images")
        logger.error('exiting...')
        sys.exit(0)

    if not math.isclose(1, sum(args.style_weights)):
        logger.error("ERROR: total style weights does not add up to 1")
        logger.error('exiting...')
        sys.exit(0)

    term = logging.StreamHandler()
    term.setLevel(logging.INFO)
    term.setFormatter(formatter)
    logger.addHandler(term)

    enable_cuda = torch.cuda.is_available() if args.backend == 'cuda' else False
    if args.input_size is None:
        target_shape = (512, 512) if enable_cuda else (224, 224)
        logger.info('target shape not set by user, set to {} x {}'.format(*target_shape))
    else:
        target_shape(args.input_size, args.input_size)


    logger.info('fetching vgg19 model')
    cnn = torchvision.models.vgg19(pretrained=True).features
    if enable_cuda:
         cnn.cuda()
    logger.info('initializing model')
    stylize = NeuralStylizer(cnn, args.content_image, args.style_images, args.style_weights,
                             target_shape, args.backend, args.content_layers, args.style_layers, args.pooling,
                             args.A, args.B, args.L)
    logger.info('optimizing..')
    stylize(args.iterations, args.output_size, args.init, args.output_file)
    logger.info('complete!')
    logger.info('saving output to {}'.format(args.output_file))
main()
