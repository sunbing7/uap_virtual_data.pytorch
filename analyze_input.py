from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn

from utils.data import get_data_specs, get_data, get_data_class
from utils.utils import get_model_path, get_result_path, get_uap_path, get_attribution_path, get_attribution_name
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import solve_input_attribution
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg

from matplotlib import pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis on Input')
    parser.add_argument('--causal_type', default='logit', choices=['logit', 'act', 'slogit', 'sact', 'uap_act', 'inact', 'be_act'],
                        help='Causality analysis type (default: logit)')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Used dataset to generate UAP (default: cifar10)')

    parser.add_argument('--arch', default='alexnet', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet56',
                                                                       'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                                       'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                                       'inception_v3'])
    parser.add_argument('--model_name', type=str, default='vgg19_cifar10.pth',
                        help='model name (default: vgg19_cifar10.pth)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')

    parser.add_argument('--split_layer', type=int, default=43,
                        help='causality analysis layer (default: 43)')
    # Parameters regarding UAP
    parser.add_argument('--num_iterations', type=int, default=32,
                        help='Number of iterations for causality analysis (default: 32)')
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')

    parser.add_argument('--targeted',  type=bool, default='',
                        help='Target a specific class (default: False)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Target class (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    return args


def main():
    args = parse_arguments()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # get the result path to store the results
    result_path = get_result_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed,
                                result_subfolder=args.result_subfolder,
                                postfix=args.postfix)

    print('save path : {}'.format(result_path))
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print("{} : {}".format(key, value))
    print("Random Seed: {}".format(args.seed))
    print("Python version : {}".format(sys.version.replace('\n', ' ')))
    print("Torch  version : {}".format(torch.__version__))
    print("Cudnn  version : {}".format(torch.backends.cudnn.version()))

    _, data_test = get_data(args.dataset, args.dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    #load uap
    uap = None
    if args.causal_type != 'inact':
        uap_path = get_uap_path(uap_data=args.dataset,
                                model_data=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
        uap_fn = os.path.join(uap_path, 'uap.npy')
        uap = np.load(uap_fn)
        uap = torch.from_numpy(uap)


    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
    '''
    data_train, _ = get_data(args.dataset, args.dataset)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    '''
    ####################################
    # Init model, criterion, and optimizer
    print("=> Creating model '{}'".format(args.arch))
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    print("=> Network :\n {}".format(network))

    # Set the target model into evaluation mode
    network.eval()

    # Imagenet models use the pretrained pytorch weights
    if args.dataset != "imagenet":
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)
    total_params = get_num_parameters(network)

    print("Filter Network Total # parameters: {}".format(total_params))

    if args.use_cuda:
        network.cuda()

    # perform causality analysis
    attribution_map = solve_input_attribution(data_test_loader, network, uap,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  use_cuda=args.use_cuda)

    attribution_path = get_attribution_path()
    uap_fn = os.path.join(attribution_path, "uap_attribution.npy")
    np.save(uap_fn, attribution_map)


    _, data_test = get_data_class(args.dataset, 1)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    attribution_map = solve_input_attribution(data_test_loader, network, None,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  use_cuda=args.use_cuda)
    uap_fn = os.path.join(attribution_path, "clean_attribution.npy")
    np.save(uap_fn, attribution_map)
    return


if __name__ == '__main__':
    main()
