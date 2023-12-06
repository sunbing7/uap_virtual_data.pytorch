from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn

from utils.data import get_data_specs, get_data, get_data_class, fix_labels_nips, fix_labels
from utils.utils import get_model_path, get_result_path, get_uap_path, get_attribution_path, get_attribution_name
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import solve_input_attribution, solve_input_attribution_single, solve_causal, solve_causal_single, \
    my_test, my_test_uap
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg
from causal_analysis import calculate_shannon_entropy, calculate_ssim, calculate_shannon_entropy_array
from matplotlib import pyplot as plt
from activation_analysis import outlier_detection
from utils.training import train_repair, metrics_evaluate_test, adv_train

import warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis on Input')
    parser.add_argument('--option', default='analyze_inputs', choices=['analyze_inputs', 'calc_entropy',
                                                                       'analyze_layers', 'calc_pcc', 'analyze_clean',
                                                                       'test', 'pcc', 'entropy', 'classify', 'repair_ae',
                                                                       'repair'],
                        help='Run options')
    parser.add_argument('--causal_type', default='logit', choices=['logit', 'act', 'slogit', 'sact', 'uap_act', 'inact', 'be_act'],
                        help='Causality analysis type (default: logit)')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Used dataset to generate UAP (default: cifar10)')
    parser.add_argument('--is_train', type=int, default=0)
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
    parser.add_argument('--split_layers', type=int, nargs="*", default=[43])
    # Parameters regarding UAP
    parser.add_argument('--num_iterations', type=int, default=32,
                        help='Number of iterations for causality analysis (default: 32)')
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')

    parser.add_argument('--idx', type=int, default=0)

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

    parser.add_argument('--analyze_clean', type=int, default=0)

    parser.add_argument('--th', type=float, default=2)

    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--ae_alpha', type=float, default=0.5)
    parser.add_argument('--ae_iter', type=int, default=10)

    parser.add_argument('--is_nips', default=1, type=int,
                        help='Evaluation on NIPS data')

    parser.add_argument('--loss_function', default='ce', choices=['ce', 'neg_ce', 'logit', 'bounded_logit',
                                                                  'bounded_logit_fixed_ref', 'bounded_logit_neg'],
                        help='Used loss function for source classes: (default: bounded_logit_fixed_ref)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate (default: 0.001)')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--epsilon', type=float, default=0.03922,
                        help='Norm restriction of UAP (default: 10/255)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
    print('use_cuda: {}'.format(args.use_cuda))
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    return args


def analyze_inputs(args):

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
    print("Random Seed: {}".format(args.seed))
    print("Python version : {}".format(sys.version.replace('\n', ' ')))
    print("Torch  version : {}".format(torch.__version__))
    print("Cudnn  version : {}".format(torch.backends.cudnn.version()))

    _, data_test = get_data(args.dataset, args.dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
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
    #print("=> Creating model '{}'".format(args.arch))
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
    attribution_map = solve_input_attribution_single(data_test_loader, network, uap,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  use_cuda=args.use_cuda)

    #save multiple maps
    attribution_path = get_attribution_path()
    for i in range(0, len(attribution_map)):
        attribution_map_ = attribution_map[i]
        uap_fn = os.path.join(attribution_path, "uap_attribution_single_" + str(i) + ".npy")
        np.save(uap_fn, attribution_map_)

    _, data_test = get_data_class(args.dataset, 1)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
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

    # single attribution on clean samples
    attribution_map = solve_input_attribution_single(data_test_loader, network, None,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  use_cuda=args.use_cuda)

    #save multiple maps
    attribution_path = get_attribution_path()
    for i in range (0, len(attribution_map)):
        attribution_map_ = attribution_map[i]
        uap_fn = os.path.join(attribution_path, "clean_attribution_single_" + str(i) + ".npy")
        np.save(uap_fn, attribution_map_)
    return


def analyze_layers(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
    ####################################
    # Init model, criterion, and optimizer
    #print("=> Creating model '{}'".format(args.arch))
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                          input_size=input_size,
                          num_classes=num_classes,
                          finetune=False)

    #print("=> Network :\n {}".format(network))

    # Set the target model into evaluation mode
    network.eval()

    # Imagenet models use the pretrained pytorch weights
    if args.dataset != "imagenet":
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)
    total_params = get_num_parameters(network)

    #print("Filter Network Total # parameters: {}".format(total_params))

    if args.use_cuda:
        network.cuda()

    uap = None
    if not args.analyze_clean:
        _, data_test = get_data(args.dataset, args.dataset)

        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True)
        #load uap
        if args.causal_type != 'inact':
            uap_path = get_uap_path(uap_data=args.dataset,
                                    model_data=args.dataset,
                                    network_arch=args.arch,
                                    random_seed=args.seed)
            uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
            uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
            uap = torch.from_numpy(uap)

        # perform causality analysis
        attribution_map, outputs, clean_outputs = solve_causal_single(data_test_loader, network, uap, args.arch,
                                      split_layer=args.split_layer,
                                      targeted=args.targeted,
                                      target_class=args.target_class,
                                      num_sample=args.num_iterations,
                                      causal_type=args.causal_type,
                                      log=None,
                                      use_cuda=args.use_cuda)

        #save multiple maps
        attribution_path = get_attribution_path()
        for i in range(0, len(attribution_map)):
            attribution_map_ = attribution_map[i]
            uap_fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + "_s" + str(i)
                                  + '_' + str(outputs[i]) + ".npy")
            np.save(uap_fn, attribution_map_)
        output_fn = os.path.join(attribution_path, "uap_clean_outputs_" + str(args.split_layer) + ".npy")
        np.save(output_fn, clean_outputs)
    else:
        data_train, data_test = get_data_class(args.dataset, args.target_class)
        if len(data_train) == 0:
            print('No sample from class {}'.format(args.target_class))
            return
        data_test_loader = torch.utils.data.DataLoader(data_train,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True)

        attribution_map, outputs, clean_outputs = solve_causal_single(data_test_loader, network, None, args.arch,
                                      split_layer=args.split_layer,
                                      targeted=args.targeted,
                                      target_class=args.target_class,
                                      num_sample=args.num_iterations,
                                      causal_type=args.causal_type,
                                      log=None,
                                      use_cuda=args.use_cuda)

        attribution_path = get_attribution_path()
        for i in range(0, len(attribution_map)):
            attribution_map_ = attribution_map[i]
            uap_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + "_s" + str(i)
                                  + '_' + str(outputs[i]) + ".npy")
            np.save(uap_fn, attribution_map_)
        output_fn = os.path.join(attribution_path, "clean_outputs_" + str(args.split_layer) + ".npy")
        np.save(output_fn, clean_outputs)
    return


def analyze_layers_clean(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    #load uap
    uap = None

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    ####################################
    # Init model, criterion, and optimizer
    #print("=> Creating model '{}'".format(args.arch))
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    #print("=> Network :\n {}".format(network))

    # Set the target model into evaluation mode
    network.eval()

    # Imagenet models use the pretrained pytorch weights
    if args.dataset != "imagenet":
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)
    total_params = get_num_parameters(network)

    #print("Filter Network Total # parameters: {}".format(total_params))

    if args.use_cuda:
        network.cuda()

    data_train, _ = get_data_class(args.dataset, args.target_class)
    #print('Number of training samples in this class: {}'.format(len(data_train)))

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    attribution_map = solve_causal(data_train_loader, network, None, args.arch,
                                  split_layer=args.split_layer,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  log=None,
                                  use_cuda=args.use_cuda)

    attribution_path = get_attribution_path()
    file_name = "clean_attribution_" + str(args.split_layer) + '_' + str(args.target_class) + "_avg.npy"
    uap_fn = os.path.join(attribution_path, file_name)
    np.save(uap_fn, attribution_map)

    return


def calc_entropy(args):
    print('idx is {}'.format(args.idx))
    attribution_path = get_attribution_path()
    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer)
                            + '_' + str(args.target_class) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    if not args.analyze_clean:
        fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + '_' + str(args.target_class) + ".npy")
    else:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + '_' + str(args.target_class) + ".npy")

    loaded = np.load(fn)

    if args.causal_type == 'logit':
        uap_ca = loaded[:, 1]
    elif args.causal_type == 'act':
        uap_ca = loaded.transpose()

    clean_h = calculate_shannon_entropy_array(clean_ca)
    uap_h = calculate_shannon_entropy_array(uap_ca)

    print('entropy avg, single: {} {}'.format(clean_h, uap_h))


def calc_entropy_i(i, args):
    attribution_path = get_attribution_path()

    if not args.analyze_clean:
        fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                          str(i) + '_' + str(args.target_class) + ".npy")
    else:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(i) + '_' + str(args.target_class) + ".npy")
    if os.path.exists(fn):
        loaded = np.load(fn)
    else:
        return

    if args.causal_type == 'logit':
        ca = loaded[:, 1]
    elif args.causal_type == 'act':
        ca = loaded.transpose()

    #clean_h = calculate_shannon_entropy_array(clean_ca)
    uap_h = calculate_shannon_entropy_array(ca)

    #print('entropy {}: {}'.format(i, uap_h))
    return uap_h

def calc_entropy_old():
    attribution_path = get_attribution_path()
    uap_fn = os.path.join(attribution_path, "uap_attribution_s_4.npy")
    loaded = np.load(uap_fn)
    uap_ca = np.transpose(loaded[:, 1].reshape(3, 224, 224), (1, 2, 0))
    #uap_ca = uap_ca[:, :, 2]
    uap_h = calculate_shannon_entropy(uap_ca, 224*224*3)

    clean1_fn = os.path.join(attribution_path, "clean_attribution_s_4.npy")
    loaded = np.load(clean1_fn)
    clean1_ca = np.transpose(loaded[:, 1].reshape(3, 224, 224), (1, 2, 0))
    clean1_h = calculate_shannon_entropy(clean1_ca, 224*224*3)

    clean_fn = os.path.join(attribution_path, "clean_attribution.npy")
    loaded = np.load(clean_fn)
    clean_ca = np.transpose(loaded[:, -1].reshape(3, 224, 224), (1, 2, 0))
    #clean_ca = clean_ca[:, :, 2]
    clean_h = calculate_shannon_entropy(clean_ca, 224*224*3)

    print('uap_h: {}, clean1_h: {}, clean_h: {}'.format(uap_h, clean1_h, clean_h))
    print('entropy difference uap vs clean: {}'.format((uap_h - clean_h) / (clean_h)))
    print('entropy difference clean1 vs clean: {}'.format((clean1_h - clean_h) / (clean_h)))

    ssim = calculate_ssim(uap_ca, clean_ca)
    print("Image similarity uap vs clean: {}".format(ssim))

    ssim = calculate_ssim(clean1_ca, clean_ca)
    print("Image similarity clean1 vs clean: {}".format(ssim))
    return uap_h, clean_h, ssim


def calc_entropy_layer_old():
    attribution_path = get_attribution_path()
    uap_fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + "_s7.npy")
    loaded = np.load(uap_fn)
    uap_ca = loaded[:, 1]

    uap_h = calculate_shannon_entropy_array(uap_ca)

    clean1_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + "_s7.npy")
    loaded = np.load(clean1_fn)
    clean1_ca = loaded[:, 1]
    clean1_h = calculate_shannon_entropy_array(clean1_ca)

    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    clean_h = calculate_shannon_entropy_array(clean_ca)

    print('uap_h: {}, clean1_h: {}, clean_h: {}'.format(uap_h, clean1_h, clean_h))
    print('entropy difference uap vs clean: {}'.format((uap_h - clean_h) / (clean_h)))
    print('entropy difference clean1 vs clean: {}'.format((clean1_h - clean_h) / (clean_h)))

    return


def calc_entropy_layer(i):
    attribution_path = get_attribution_path()
    uap_fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + "_s" + str(i) + ".npy")
    loaded = np.load(uap_fn)
    uap_ca = loaded[:, 1]

    uap_h = calculate_shannon_entropy_array(uap_ca)

    clean1_fn = os.path.join(attribution_path,
                             "clean_attribution_" + str(args.split_layer) + "_s" + str(i) + ".npy")
    loaded = np.load(clean1_fn)
    clean1_ca = loaded[:, 1]
    clean1_h = calculate_shannon_entropy_array(clean1_ca)
    print('uap_h: {}, clean1_h: {}'.format(uap_h, clean1_h))


def calc_pcc(args):
    #print('idx is {}'.format(args.idx))
    attribution_path = get_attribution_path()
    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer)
                            + '_' + str(args.target_class) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    if not args.analyze_clean:
        fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + '_' + str(args.target_class) + ".npy")
        prefix = 'uap'
    else:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + '_' + str(args.target_class) + ".npy")
        prefix = 'clean'

    if os.path.exists(fn):
        loaded = np.load(fn)
    else:
        return

    if args.causal_type == 'logit':
        uap_ca = loaded[:, 1]
    elif args.causal_type == 'act':
        uap_ca = loaded.transpose()

    uap_pcc = np.corrcoef(uap_ca, clean_ca)[0, 1]

    print('{} pcc: {}'.format(prefix, uap_pcc))

    return uap_pcc


def calc_pcc_i(i, args):
    attribution_path = get_attribution_path()
    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer)
                            + '_' + str(args.target_class) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    if not args.analyze_clean:
        fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                          str(i) + '_' + str(args.target_class) + ".npy")
    else:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(i) + '_' + str(args.target_class) + ".npy")
    if os.path.exists(fn):
        loaded = np.load(fn)
    else:
        return

    if args.causal_type == 'logit':
        ca = loaded[:, 1]
    elif args.causal_type == 'act':
        ca = loaded.transpose()

    uap_pcc = np.corrcoef(ca, clean_ca)[0, 1]

    #print('pcc {}: {}'.format(i, uap_pcc))
    return uap_pcc


def calc_pcc_i_old(i, args):
    attribution_path = get_attribution_path()

    clean_fn = os.path.join(attribution_path, args.avg_ca_name)
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    uap_fn = os.path.join(attribution_path, args.uap_ca_name)
    loaded = np.load(uap_fn)
    uap_ca = loaded[:, 1]

    uap_pcc = np.corrcoef(uap_ca, clean_ca)[0, 1]

    clean1_fn = os.path.join(attribution_path, args.clean_ca_name)
    loaded = np.load(clean1_fn)
    clean1_ca = loaded[:, 1]
    clean1_pcc = np.corrcoef(clean1_ca, clean1_ca)[0, 1]
    print('{}: {}, {}'.format(i, uap_pcc, clean1_pcc))


'''
def rearrange_outputfile():
    attribution_path = get_attribution_path()
    ca_map = []
    for i in range(0, 224*224*3):
        fn = os.path.join(attribution_path, "uap_attribution_single_" + str(i) + ".npy")
        ca_map.append(np.load(fn))
        os.rename(fn, os.path.join(attribution_path, 'backup', "uap_attribution_single_" + str(i) + ".npy"))
    ca_map = np.array(ca_map)
    ca_map = np.transpose(ca_map, (1, 0, 2))

    for i in range(0, len(ca_map)):
        attribution_map_ = ca_map[i]
        uap_fn = os.path.join(attribution_path, "uap_attribution_s_" + str(i) + ".npy")
        np.save(uap_fn, attribution_map_)

    ca_map = []
    for i in range(0, 224*224*3):
        fn = os.path.join(attribution_path, "clean_attribution_single_" + str(i) + ".npy")
        ca_map.append(np.load(fn))
        os.rename(fn, os.path.join(attribution_path, 'backup', "clean_attribution_single_" + str(i) + ".npy"))
    ca_map = np.array(ca_map)
    ca_map = np.transpose(ca_map, (1, 0, 2))

    for i in range(0, len(ca_map)):
        attribution_map_ = ca_map[i]
        uap_fn = os.path.join(attribution_path, "clean_attribution_s_" + str(i) + ".npy")
        np.save(uap_fn, attribution_map_)

    return
'''


def test(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    #load uap
    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
    uap = np.load(uap_fn)
    uap = torch.from_numpy(uap)

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    ####################################
    # Init model, criterion, and optimizer
    #print("=> Creating model '{}'".format(args.arch))
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

    if args.use_cuda:
        network.cuda()

    data_train, data_test = get_data(args.dataset, args.dataset)

    if args.is_train:
        if args.dataset == "imagenet":
            data_train = fix_labels(data_train)
        dataset = data_train
    else:
        # Fix labels if needed
        if args.dataset == "imagenet":
            data_test = fix_labels_nips(data_test, pytorch=True)
        dataset = data_test


    data_test_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    _, acc, _, fr, _ = my_test_uap(data_test_loader, network, uap, args.batch_size, args.num_iterations, split_layer=43,
                      use_cuda=args.use_cuda)
    print('overall acc {}'.format(acc))
    print('overall fooling ratio {}'.format(fr))

    tot_correct = 0
    tot_num = 0
    for cur_class in range(0, 1000):
        data_train, data_test = get_data_class(args.dataset, cur_class)
        if len(data_test) == 0:
            continue
        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True)

        corr, _, fool, _, num = my_test_uap(data_test_loader, network, uap, args.batch_size, args.num_iterations, split_layer=43,
                               use_cuda=args.use_cuda)
        print('class {}, correct {}, fool {}, num {}'.format(cur_class, corr, fool, num))
        tot_correct += corr
        tot_num += num
    print('Model accuracy: {}%'.format(tot_correct / tot_num * 100))

    return


def process_pcc(args):
    uap_pccs = []
    clean_pccs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            print('uap_pcc: {}'.format(uap_pcc))

    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            print('clean_pcc: {}'.format(clean_pcc))
    return


def process_entropy(args):
    uap_hs = []
    clean_hs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            print('uap_h: {}'.format(uap_h))

    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            print('clean_h: {}'.format(clean_h))
    return


def uap_classification_avg(args):
    #get average entropy of clean data
    clean_hs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            #print('clean_h: {}'.format(clean_h))
    clean_hs_avg = np.mean(np.array(clean_hs))

    #get entropy of the test sample
    h_result = []
    uap_hs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            h_result.append(int(uap_h > clean_hs_avg))
            #print('uap_h: {}'.format(uap_h))
    print('Layer {} entropy result[{}]: {}'.format(args.split_layer, len(h_result), h_result))

    #get average pcc of clean data
    clean_pccs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            #print('clean_pcc: {}'.format(clean_pcc))
    clean_pcc_avg = np.mean(np.array(clean_pccs))

    #get pcc of the test sample
    pcc_result = []
    uap_pccs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            pcc_result.append(int(uap_pcc < clean_pcc_avg))
            #print('uap_pcc: {}'.format(uap_pcc))
    print('Layer {} pcc result[{}]    : {}'.format(args.split_layer, len(pcc_result), pcc_result))
    return np.sum(np.logical_and(np.array(h_result) == 1, np.array(pcc_result) == 1)) / len(pcc_result) * 100


def uap_classification(args):
    #get average entropy of clean data
    clean_hs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            #print('clean_h: {}'.format(clean_h))
    clean_hs_avg = np.mean(np.array(clean_hs))

    #get entropy of the test sample
    h_result = []
    uap_hs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            top = outlier_detection((clean_hs + [uap_h]), max(clean_hs + [uap_h]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            h_result.append(int((len(clean_hs + [uap_h]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_hs + [uap_h]) - 1))
    print('Layer {} entropy result[{}]: {}'.format(args.split_layer, len(h_result), h_result))

    #get average pcc of clean data
    clean_pccs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            #print('clean_pcc: {}'.format(clean_pcc))
    clean_pcc_avg = np.mean(np.array(clean_pccs))

    #get pcc of the test sample
    pcc_result = []
    uap_pccs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            reversed_list = max(clean_pccs + [uap_pcc]) - np.array(clean_pccs + [uap_pcc])
            top = outlier_detection(reversed_list, max(clean_pccs + [uap_pcc]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            pcc_result.append(int((len(clean_pccs + [uap_pcc]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_pccs + [uap_pcc]) - 1))
    print('Layer {} pcc result[{}]    : {}'.format(args.split_layer, len(pcc_result), pcc_result))
    return np.sum(np.logical_and(np.array(h_result) == 1, np.array(pcc_result) == 1)) / len(pcc_result) * 100


def uap_repair(args):
    _, data_test = get_data(args.dataset, args.dataset)
    # Fix labels if needed
    if args.is_nips:
        print('is_nips')
        data_test = fix_labels_nips(data_test, pytorch=True)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    data_train, _ = get_data(args.dataset, args.dataset)

    if args.dataset == "imagenet":
        data_train = fix_labels(data_train)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
    uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
    uap = torch.from_numpy(uap)

    ####################################
    # Init model, criterion, and optimizer
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    target_network = get_network(args.arch,
                                 input_size=input_size,
                                 num_classes=num_classes,
                                 finetune=False)

    # Imagenet models use the pretrained pytorch weights
    if args.dataset != "imagenet":
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    #non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print("Target Network Trainable parameters: {}".format(trainale_params))
    print("Target Network Total # parameters: {}".format(total_params))

    target_network.train()

    if args.loss_function == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_function == "neg_ce":
        criterion = NegativeCrossEntropy()
    elif args.loss_function == "logit":
        criterion = LogitLoss(num_classes=num_classes, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit":
        criterion = BoundedLogitLoss(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_fixed_ref":
        criterion = BoundedLogitLossFixedRef(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_neg":
        criterion = BoundedLogitLoss_neg(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    else:
        raise ValueError

    print('Criteria: {}'.format(criterion))

    if args.use_cuda:
        target_network.cuda()
        criterion.cuda()

    optimizer = torch.optim.SGD(target_network.parameters(), lr=args.learning_rate, momentum=0.9)
    #'''
    # Measure the time needed for the UAP generation
    start = time.time()

    if 'ae' in args.option:
        adv_train(data_train_loader,
                  target_network,
                  args.arch,
                  criterion,
                  optimizer,
                  args.num_iterations,
                  args.split_layers,
                  alpha=args.alpha,
                  ae_alpha=args.ae_alpha,
                  print_freq=args.print_freq,
                  use_cuda=args.use_cuda,
                  adv_itr=args.ae_iter,
                  eps=args.epsilon)
    else:
        train_repair(data_loader=data_train_loader,
                         model=target_network,
                         arch=args.arch,
                         criterion=criterion,
                         optimizer=optimizer,
                         num_iterations=args.num_iterations,
                         split_layers=args.split_layers,
                         alpha=args.alpha,
                         print_freq=args.print_freq,
                         use_cuda=args.use_cuda)

    end = time.time()
    print("Time needed for UAP repair: {}".format(end - start))

    #eval
    if args.use_cuda:
        uap = uap.cuda()
    metrics_evaluate_test(data_loader=data_test_loader,
                          target_model=target_network,
                          uap=uap,
                          targeted=args.targeted,
                          target_class=args.target_class,
                          log=None,
                          use_cuda=args.use_cuda)


def clean_classification(args):
    #get average entropy of clean data
    clean_hs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            #print('clean_h: {}'.format(clean_h))
    clean_hs_avg = np.mean(np.array(clean_hs))

    #get entropy of the test sample
    h_result = []
    uap_hs = []
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            reversed_list = max(clean_hs + [uap_h]) - np.array(clean_hs + [uap_h])
            top = outlier_detection(reversed_list, max(clean_hs + [uap_h]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            h_result.append(int((len(clean_hs + [uap_h]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_hs + [uap_h]) - 1))
    print('Layer {} entropy result[{}]: {}'.format(args.split_layer, len(h_result), h_result))

    #get average pcc of clean data
    clean_pccs = []
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            #print('clean_pcc: {}'.format(clean_pcc))
    clean_pcc_avg = np.mean(np.array(clean_pccs))

    #get pcc of the test sample
    pcc_result = []
    uap_pccs = []
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            top = outlier_detection((clean_pccs + [uap_pcc]), max(clean_pccs + [uap_pcc]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            pcc_result.append(int((len(clean_pccs + [uap_pcc]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_pccs + [uap_pcc]) - 1))
    print('Layer {} pcc result[{}]    : {}'.format(args.split_layer, len(pcc_result), pcc_result))
    return np.sum(np.logical_and(np.array(h_result) == 1, np.array(pcc_result) == 1) / len(pcc_result)) * 100


if __name__ == '__main__':
    args = parse_arguments()
    state = {k: v for k, v in args._get_kwargs()}
    start = time.time()
    '''
    attribution_path = get_attribution_path()
    output_fn = os.path.join(attribution_path, "uap_clean_outputs_" + str(args.split_layer) + ".npy")
    clean_outputs = np.load(output_fn)
    print(clean_outputs)


    for key, value in state.items():
        print("{} : {}".format(key, value))
    '''
    if args.option == 'analyze_inputs':
        analyze_inputs(args)
    elif args.option == 'calc_entropy':
        if args.num_iterations != 0:
            for i in range(0, args.num_iterations):
                calc_entropy_i(i, args)
        else:
            calc_entropy(args)
    elif args.option == 'calc_pcc':
        if args.num_iterations != 0:
            for i in range(0, args.num_iterations):
                calc_pcc_i(i, args)
        else:
            calc_pcc(args)
    elif args.option == 'analyze_layers':
        analyze_layers(args)
    elif args.option == 'analyze_clean':
        analyze_layers_clean(args)
    elif args.option == 'test':
        test(args)
    elif args.option == 'pcc':
        process_pcc(args)
    elif args.option == 'entropy':
        process_entropy(args)
    elif args.option == 'classify':
        tpr = uap_classification(args)
        fpr = clean_classification(args)
        print('TPR: {}, FPR: {}'.format(tpr, fpr))
    elif 'repair' in args.option:
        uap_repair(args)
    end = time.time()
    #print('Process time: {}'.format(end - start))

