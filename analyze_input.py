from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn

from utils.data import get_data_specs, get_data, get_data_class, fix_labels_nips
from utils.utils import get_model_path, get_result_path, get_uap_path, get_attribution_path, get_attribution_name
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import solve_input_attribution, solve_input_attribution_single, solve_causal, solve_causal_single, \
    my_test
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg
from causal_analysis import calculate_shannon_entropy, calculate_ssim, calculate_shannon_entropy_array
from matplotlib import pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis on Input')
    parser.add_argument('--option', default='analyze_inputs', choices=['analyze_inputs', 'calc_entropy',
                                                                       'analyze_layers', 'calc_pcc', 'analyze_clean',
                                                                       'test'],
                        help='Run options')
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

    parser.add_argument('--analyze_clean', type=int, default=0)

    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

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
            uap = np.load(uap_fn)
            uap = torch.from_numpy(uap)

        # perform causality analysis
        attribution_map, outputs = solve_causal_single(data_test_loader, network, uap, args.arch,
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

    else:
        _, data_test = get_data_class(args.dataset, args.target_class)
        if len(data_test) == 0:
            print('No sample from class {}'.format(args.target_class))
            return
        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True)

        attribution_map, outputs = solve_causal_single(data_test_loader, network, None, args.arch,
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

    data_train, _ = get_data_class(args.dataset, args.target_class)
    print('Number of training samples in this class: {}'.format(len(data_train)))

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


def calc_entropy():
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


def calc_entropy_layer():
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


def calc_entropy_pcc(i, args):
    attribution_path = get_attribution_path()

    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    uap_fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + "_s" + str(i) + ".npy")
    loaded = np.load(uap_fn)
    uap_ca = loaded[:, 1]

    uap_pcc = np.corrcoef(uap_ca, clean_ca)[0, 1]

    clean1_fn = os.path.join(attribution_path,
                             "clean_attribution_" + str(args.split_layer) + "_s" + str(i) + ".npy")
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
    uap = None

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

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

    if args.use_cuda:
        network.cuda()

    _, data_test = get_data(args.dataset, args.dataset)

    # Fix labels if needed
    data_test = fix_labels_nips(data_test, pytorch=True)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    _, acc, _ = my_test(data_test_loader, network, uap, args.batch_size, args.num_iterations, split_layer=43,
                      use_cuda=args.use_cuda)
    print('overall acc {}'.format(acc))

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

        corr, _, num = my_test(data_test_loader, network, uap, args.batch_size, args.num_iterations, split_layer=43,
                               use_cuda=args.use_cuda)
        print('class {}, correct {}, num {}'.format(cur_class, corr, num))
        tot_correct += corr
        tot_num += num
    print('Model accuracy: {}%'.format(tot_correct / tot_num * 100))

    return


if __name__ == '__main__':
    args = parse_arguments()
    state = {k: v for k, v in args._get_kwargs()}
    start = time.time()
    for key, value in state.items():
        print("{} : {}".format(key, value))
    if args.option == 'analyze_inputs':
        analyze_inputs(args)
    elif args.option == 'calc_entropy':
        calc_entropy_layer()
    elif args.option == 'calc_pcc':
        for i in range(0, args.num_iterations):
            calc_entropy_pcc(i, args)
    elif args.option == 'analyze_layers':
        analyze_layers(args)
    elif args.option == 'analyze_clean':
        analyze_layers_clean(args)
    elif args.option == 'test':
        test(args)
    end = time.time()
    print('Process time: {}'.format(end - start))

