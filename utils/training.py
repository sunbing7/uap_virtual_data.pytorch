from __future__ import division
import numpy as np
import os, shutil, time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.utils import time_string, print_log
from torch.distributions import Categorical
from torch.autograd import Variable
from torchsummary import summary
from matplotlib import pyplot as plt


DEBUG = True

def calculate_entropy_tensor(x):
    """
    calculate information entropy
    Returns:
    H
    """
    entropy = Categorical(probs=x).entropy()
    return entropy


def calculate_shannon_entropy_array(x):
    """
    calculate information entropy
    Returns:
    H
    """
    x = np.array(x)

    x = x.flatten(order='C')
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    h = -np.sum(probabilities * np.log2(probabilities))

    return h


def calculate_shannon_entropy_batch(x):
    """
    calculate information entropy
    Returns:
    H
    """
    x = np.array(x)
    out = np.zeros(x.shape)
    for index, x_i in enumerate(x):
        _, counts = np.unique(x_i, return_counts=True)
        probabilities = counts / x.shape[-1]
        product = probabilities * np.log2(probabilities)
        out[index][:len(counts)] = product

    h = -np.sum(out, axis=-1)
    return h


def train(data_loader,
          model,
          criterion,
          optimizer,
          epsilon,
          num_iterations,
          targeted,
          target_class,
          log,
          print_freq=200,
          use_cuda=True):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.module.generator.train()
    model.module.target_model.eval()

    end = time.time()

    data_iterator = iter(data_loader)

    iteration=0
    while iteration < num_iterations:
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)

        if targeted:
            target = torch.ones(input.shape[0], dtype=torch.int64) * target_class
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        if model.module._get_name() == "Inception3":
            output, aux_output = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4*loss2
        else:
            output = model(input)
            loss = criterion(output, target)

        if not targeted:
            loss = -loss

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projection
        model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        iteration, num_iterations, batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

        iteration+=1
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100-top1.avg), log)


def train_advanced(data_loader,
                   model,
                   arch,
                   criterion,
                   optimizer,
                   epsilon,
                   num_iterations,
                   split_layers,
                   targeted,
                   target_class,
                   log,
                   print_freq=200,
                   use_cuda=True,
                   en_weight=0.5,
                   adjust=5.0):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.module.generator.train()
    model.module.target_model.eval()

    end = time.time()

    data_iterator = iter(data_loader)

    en_loss = Variable(torch.tensor(.0), requires_grad=True)
    en_cri = hloss(use_cuda)

    iteration=0
    while iteration < num_iterations:
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)

        pmodel1, pmodel2 = split_perturbed_model(model, arch, split_layer=split_layers[0])

        if targeted:
            target = torch.ones(input.shape[0], dtype=torch.int64) * target_class
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        if model.module._get_name() == "Inception3":
            output, aux_output = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4*loss2
        else:
            #output_ori = model(input)
            poutput = pmodel1(input)
            output = pmodel2(poutput)
            en_loss = torch.mean(en_cri(poutput.view(len(input), -1)))
            lcce = criterion(output, target)
            #print('[DEBUG] lcce: {}, en_loss: {}'.format(lcce, en_loss))
            loss = (1.0 - en_weight) * lcce - en_weight * adjust * en_loss

        if not targeted:
            loss = -loss

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projection
        model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        iteration, num_iterations, batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

        iteration+=1
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100-top1.avg), log)


def train_hidden(data_loader,
                  model,
                  model2,
                  criterion,
                  optimizer,
                  epsilon,
                  num_iterations,
                  targeted,
                  target_class,
                  log,
                  arch,
                  mask,
                  split_layer,
                  do_val=1,
                  num_hidden_neu=4096,
                  print_freq=200,
                  use_cuda=True):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.module.generator.train()
    model.module.target_model.eval()

    end = time.time()

    data_iterator = iter(data_loader)

    iteration = 0
    while (iteration < num_iterations):
        try:
            input, target = next(data_iterator)
            target_ = np.ones((len(target), num_hidden_neu)) * do_val
            masks = np.tile(mask, (len(input), 1))
            target = torch.from_numpy(target_ * masks).long()
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)

        if targeted:
            target_ = torch.ones(input.shape[0], dtype=torch.int64) * target_class
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        if model.module._get_name() == "Inception3":
            output, aux_output = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4 * loss2
        else:
            output = model(input)
            if output.shape != target.shape:
                #print(iteration)
                #print(input.shape)
                #print(output.shape)
                #print(target.shape)
                target = torch.reshape(target, output.shape)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if len(target_.shape) > 1:
            target_ = torch.argmax(target_, dim=-1)
        if use_cuda:
            target_ = target_.cuda()

        final_output = model2(output)
        prec1, prec5 = accuracy(final_output.data, target_, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Projection
        model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                iteration, num_iterations, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

        iteration += 1
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100 - top1.avg),
              log)


def train_repair(data_loader,
                 model,
                 arch,
                 criterion,
                 optimizer,
                 num_iterations,
                 num_batches=1000,
                 print_freq=200,
                 use_cuda=True):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    print('[DEBUG] dataloader length: {}'.format(len(data_loader)))

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()
                loss = criterion(output, target)

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/1563]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    num_batch, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
              'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
               iteration, num_iterations, batch_time=batch_time,
               data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model

def adv_train(data_loader,
              model,
              arch,
              criterion,
              optimizer,
              num_iterations,
              split_layers,
              uap=None,
              num_batches=1000,
              alpha=0.1,
              use_cuda=True,
              adv_itr=10,
              eps=0.0392,
              mean=[0, 0, 0],
              std=[1, 1, 1]):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if DEBUG:
        adv_top1 = AverageMeter()
        adv_top5 = AverageMeter()
        delta_top1 = AverageMeter()
        delta_top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    print('[DEBUG] dataloader length: {}'.format(len(data_loader)))

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break
            p_models = []
            for split_layer in split_layers:
                pmodel, _ = split_model(model, arch, split_layer=split_layer)
                p_models = p_models + [pmodel]

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()
                uap = uap.cuda()

            # generate AEs
            delta = ae_training_individual(p_models,
                                           input,
                                           adv_itr,
                                           eps,
                                           True,
                                           mean,
                                           std,
                                           use_cuda)

            x_adv = input + delta

            uap_x = (input + uap).float()

             # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                if DEBUG:
                    delta_output = model(x_adv)
                    adv_output = model(uap_x)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                poutput = model(x_adv)
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if DEBUG:
                adv_prec1, adv_prec5 = accuracy(adv_output.data, target_, topk=(1, 5))
                adv_top1.update(adv_prec1.item(), input.size(0))
                adv_top5.update(adv_prec5.item(), input.size(0))

                delta_prec1, delta_prec5 = accuracy(delta_output.data, target_, topk=(1, 5))
                delta_top1.update(delta_prec1.item(), input.size(0))
                delta_top5.update(delta_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            for pmodel in p_models:
                del pmodel

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/{}]   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      'deltaPrec@1 {delta_top1.val:.3f} ({delta_top1.avg:.3f})   '
                      'deltaPrec@5 {delta_top5.val:.3f} ({delta_top5.avg:.3f})   '
                      .format(
                    num_batch, len(data_loader),
                    loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5,
                    delta_top1=delta_top1, delta_top5=delta_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '.format(
            iteration, num_iterations,
            loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model


def pgd_train(data_loader,
              model,
              target_class,
              criterion,
              optimizer,
              num_iterations,
              uap=None,
              num_batches=1000,
              alpha=0.1,
              use_cuda=True,
              adv_itr=10,
              eps=0.0392,
              mean=[0, 0, 0],
              std=[1, 1, 1]):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()
    delta_top1 = AverageMeter()
    delta_top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()
                uap = uap.cuda()
            #targeted
            tgt_class = torch.ones_like(target) * target_class

            delta = ae_training_pgd_tgt(model,
                                        input,
                                        tgt_class,
                                        criterion,
                                        adv_itr,
                                        eps,
                                        True,
                                        mean,
                                        std,
                                        use_cuda)
            #'''
            x_adv = input + delta

            uap_x = (input + uap).float()

            # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                delta_output = model(x_adv)
                adv_output = model(uap_x)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                poutput = model(x_adv)
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            adv_prec1, adv_prec5 = accuracy(adv_output.data, target_, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

            delta_prec1, delta_prec5 = accuracy(delta_output.data, target_, topk=(1, 5))
            delta_top1.update(delta_prec1.item(), input.size(0))
            delta_top5.update(delta_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/{}]   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      'deltaPrec@1 {delta_top1.val:.3f} ({delta_top1.avg:.3f})   '
                      'deltaPrec@5 {delta_top5.val:.3f} ({delta_top5.avg:.3f})   '
                      .format(
                    num_batch, len(data_loader),
                    loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5,
                    delta_top1=delta_top1, delta_top5=delta_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '.format(
            iteration, num_iterations,
            loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model


def pgd_train_untgt(data_loader,
                    model,
                    criterion,
                    optimizer,
                    num_iterations,
                    uap=None,
                    num_batches=1000,
                    alpha=0.1,
                    use_cuda=True,
                    adv_itr=10,
                    eps=0.0392,
                    mean=[0, 0, 0],
                    std=[1, 1, 1]):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()
    delta_top1 = AverageMeter()
    delta_top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()
                uap = uap.cuda()

            #untarget
            delta = ae_training_pgd(model,
                                    input,
                                    target,
                                    criterion,
                                    adv_itr,
                                    eps,
                                    True,
                                    mean,
                                    std,
                                    use_cuda)

            x_adv = input + delta

            uap_x = (input + uap).float()

            # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                delta_output = model(x_adv)
                adv_output = model(uap_x)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                poutput = model(x_adv)
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            adv_prec1, adv_prec5 = accuracy(adv_output.data, target_, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

            delta_prec1, delta_prec5 = accuracy(delta_output.data, target_, topk=(1, 5))
            delta_top1.update(delta_prec1.item(), input.size(0))
            delta_top5.update(delta_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/{}]   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      'deltaPrec@1 {delta_top1.val:.3f} ({delta_top1.avg:.3f})   '
                      'deltaPrec@5 {delta_top5.val:.3f} ({delta_top5.avg:.3f})   '
                      .format(
                    num_batch, len(data_loader),
                    loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5,
                    delta_top1=delta_top1, delta_top5=delta_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '.format(
            iteration, num_iterations,
            loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model



def adv_ae_train( data_loader,
                  model,
                  arch,
                  criterion,
                  optimizer,
                  num_iterations,
                  split_layers,
                  uap=None,
                  std=0,
                  alpha=0.1,
                  ae_alpha=0.1,
                  print_freq=200,
                  use_cuda=True,
                  adv_itr=10,
                  eps=0.0392):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()
    delta_top1 = AverageMeter()
    delta_top5 = AverageMeter()
    # switch to train mode

    model.train()

    end = time.time()

    #print('[DEBUG] dataloader length: {}'.format(len(data_loader)))

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            p_models = []
            for split_layer in split_layers:
                pmodel, _ = split_model(model, arch, split_layer=split_layer, flat=True)
                p_models = p_models + [pmodel]

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()
                uap = uap.cuda()

            # generate AEs
            delta = ae_training_individual(model, p_models, input, target, criterion, adv_itr, eps, ae_alpha, True,
                                           use_cuda)
            # print('[DEBUG]delta.shape {}'.format(delta.shape))
            x_adv = input + delta
            uap_x = (input + uap).float()
            # x_adv = ae_training_tgt(model, input, target, criterion, adv_itr, eps, True)

            # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                delta_output = model(x_adv)
                adv_output = model(uap_x)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()
                '''
                #calculate entropy
                plosses = 0
                for pmodel in p_models:
                    poutput = pmodel(x_adv).view(len(input), -1)
                    moutput = pmodel(input).view(len(input), -1)
                    en_cri = hloss(use_cuda)
                    p_en_b = (en_cri(poutput)).mean()
                    c_en_b = (en_cri(moutput)).mean()
                '''

                poutput = model(x_adv)
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss  # - 0.001 * alpha * p_en_b
                # loss = pce_loss

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            adv_prec1, adv_prec5 = accuracy(adv_output.data, target_, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

            delta_prec1, delta_prec5 = accuracy(delta_output.data, target_, topk=(1, 5))
            delta_top1.update(delta_prec1.item(), input.size(0))
            delta_top5.update(delta_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            for pmodel in p_models:
                del pmodel

            '''
            #calculate entropy
            p_models = []
            for split_layer in split_layers:
                pmodel, _ = split_model(model, arch, split_layer=split_layer)
                p_models = p_models + [pmodel]

            for pmodel in p_models:
                poutput = pmodel(x_adv).view(len(input), -1)
                moutput = pmodel(input).view(len(input), -1)
                en_cri = hloss(use_cuda)
                p_en_aft = (en_cri(poutput)).mean()
                c_en_aft = (en_cri(moutput)).mean()

            print('[DEBUG] before hloss(poutput) {}'.format(p_en_b))
            #print('[DEBUG] after hloss(poutput) {}'.format(p_en_aft))
            #print('[DEBUG] before delta hloss(output) {}'.format(c_en_b - p_en_b))
            print('[DEBUG] clean before hloss(output) {}'.format(c_en_b))
            #print('[DEBUG] clean after hloss(output) {}'.format(c_en_aft))
            #print('[DEBUG] after delta hloss(output) {}'.format(c_en_aft - p_en_aft))

            for pmodel in p_models:
                del pmodel
            '''
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/1563]   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      'deltaPrec@1 {delta_top1.val:.3f} ({delta_top1.avg:.3f})   '
                      'deltaPrec@5 {delta_top5.val:.3f} ({delta_top5.avg:.3f})   '
                      .format(
                    num_batch,
                    loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5,
                    delta_top1=delta_top1, delta_top5=delta_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '.format(
            iteration, num_iterations,
            loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model



def gen_low_entropy_sample(data_loader,
                           model,
                           arch,
                           criterion,
                           split_layers,
                           use_cuda=True,
                           adv_itr=10,
                           eps=0.0392):

    # switch to train mode
    model.train()

    p_models = []
    for split_layer in split_layers:
        pmodel, _ = split_model(model, arch, split_layer=split_layer)
        p_models = p_models + [pmodel]

    delta = low_entropy_sample_training(model, p_models, data_loader, criterion, adv_itr, eps, True, use_cuda)

    return delta


def ae_training(model, pmodels, x, y, criterion, attack_iters=10, eps=0.0392, alpha=0.5, rs=True, use_cuda=True):
    delta = torch.zeros_like(x[0])
    if use_cuda:
        delta = delta.cuda()

    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True

    en_loss = Variable(torch.tensor(.0), requires_grad=True)
    en_cri = hloss(use_cuda)
    for i in range(attack_iters):
        delta.data = clamp(delta.data, -eps, eps)
        ae_x = torch.add(x, delta)
        output = model(ae_x)
        for pmodel in pmodels:
            poutput = pmodel(ae_x).view(len(x), -1)
            en_loss = (en_cri(poutput)).sum()
            #print('[DEBUG] itr {} ae entropy before: {}'.format(i, en_loss))
        #tgt = (torch.ones(len(ae_x), dtype=torch.int64) * 150).cuda()
        #acc_loss = criterion(output, tgt)

        #print('[DEBUG] itr {} ae acc_loss: {}'.format(i, acc_loss))
        #loss = (1 - alpha) * acc_loss - alpha * en_loss
        loss = -en_loss
        #loss = acc_loss
        loss.backward()
        grad = delta.grad.detach()

        grad_sign = sign(grad)
        delta.data = (delta + (eps / 4) * grad_sign)
        delta.data = clamp(delta.data, -eps, eps)
        delta.grad.zero_()
    '''
    for pmodel in pmodels:
        poutput = pmodel(torch.add(x, delta)).view(len(x), -1)
        en_loss = torch.mean(en_cri(poutput))
        print('[DEBUG] ae entropy after: {}'.format(en_loss))
    '''
    return delta.detach()


def ae_training_individual(pmodels,
                           x,
                           attack_iters=10,
                           eps=0.0392,
                           rs=True,
                           mean=[0, 0, 0],
                           std=[1, 1, 1],
                           use_cuda=True):
    delta = torch.zeros_like(x)

    # preprocess mean and std
    if type(mean) is not torch.Tensor:
        mean = torch.from_numpy(np.array([0, 0, 0]).reshape(1, 3, 1, 1)).float()
        std = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1)).float()

    if use_cuda:
        delta = delta.cuda()
        mean = mean.cuda()
        std = std.cuda()

    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True

    en_loss = Variable(torch.tensor(.0), requires_grad=True)
    en_cri = hloss(use_cuda)
    for i in range(attack_iters):
        delta.data = standardize_delta(clamp(delta.data, -eps, eps, use_cuda), mean, std)
        ae_x = (x + delta)
        delta.data = destandardize_delta(delta.data, mean, std)

        for pmodel in pmodels:
            poutput = pmodel(ae_x).view(len(x), -1)
            en_loss = torch.mean(en_cri(poutput))

        loss = -en_loss
        loss.backward()
        grad = delta.grad.detach()

        grad_sign = sign(grad)
        delta.data = (delta + (eps / 4) * grad_sign)
        delta.data = clamp(delta.data, -eps, eps, use_cuda)
        delta.grad.zero_()

    return standardize_delta(delta.detach(), mean, std)


def ae_training_pgd(model,
                    x,
                    y,
                    criterion,
                    attack_iters=10,
                    eps=0.0392,
                    rs=True,
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    use_cuda=True):
    '''
    adversarial training

    '''
    delta = torch.zeros_like(x)

    # preprocess mean and std
    if type(mean) is not torch.Tensor:
        mean = torch.from_numpy(np.array([0, 0, 0]).reshape(1, 3, 1, 1)).float()
        std = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1)).float()

    if use_cuda:
        delta = delta.cuda()
        mean = mean.cuda()
        std = std.cuda()

    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True

    for i in range(attack_iters):
        delta.data = standardize_delta(clamp(delta.data, -eps, eps, use_cuda), mean, std)
        ae_x = (x + delta)
        delta.data = destandardize_delta(delta.data, mean, std)

        output = model(ae_x)
        loss = criterion(output, y)

        loss.backward()
        grad = delta.grad.detach()

        grad_sign = sign(grad)
        delta.data = (delta + (eps / 4) * grad_sign)
        delta.data = clamp(delta.data, -eps, eps, use_cuda)
        delta.grad.zero_()

    return standardize_delta(delta.detach(), mean, std)


def ae_training_pgd_tgt(model,
                        x,
                        target_class,
                        criterion,
                        attack_iters=10,
                        eps=0.0392,
                        rs=True,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        use_cuda=True):
    '''
    adversarial training

    '''
    delta = torch.zeros_like(x)

    # preprocess mean and std
    if type(mean) is not torch.Tensor:
        mean = torch.from_numpy(np.array([0, 0, 0]).reshape(1, 3, 1, 1)).float()
        std = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1)).float()


    if use_cuda:
        delta = delta.cuda()
        mean = mean.cuda()
        std = std.cuda()
        target_class = target_class.cuda()

    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True

    for i in range(attack_iters):
        delta.data = standardize_delta(clamp(delta.data, -eps, eps, use_cuda), mean, std)
        ae_x = (x + delta)
        delta.data = destandardize_delta(delta.data, mean, std)

        output = model(ae_x)
        loss = criterion(output, target_class)
        #print('Iteration {} loss {}'.format(i, loss))

        loss.backward()
        grad = delta.grad.detach()

        grad_sign = sign(grad)
        delta.data = (delta - (eps / 4) * grad_sign)
        delta.data = clamp(delta.data, -eps, eps, use_cuda)
        delta.grad.zero_()

    return standardize_delta(delta.detach(), mean, std)



def low_entropy_sample_training(model, pmodels, data_loader, criterion, attack_iters=10, eps=0.0392, rs=True, use_cuda=True):
    delta = None

    en_loss = Variable(torch.tensor(.0), requires_grad=True)
    en_cri = hloss(use_cuda)
    num_samples = 0
    for x, target in data_loader:
        num_samples += len(target)
        if num_samples > 32:
            break
        for i in range(attack_iters):
            if use_cuda:
                target = target.cuda()
                x = x.cuda()
            if delta == None:
                delta = torch.zeros_like(x[0])
                if use_cuda:
                    delta = delta.cuda()

                if rs:
                    delta.uniform_(-eps, eps)

                delta.requires_grad = True

            ae_x = clamp(torch.add(x, delta), 0, 1)
            output = model(ae_x)
            for pmodel in pmodels:
                poutput = pmodel(ae_x).view(len(x), -1)
                en_loss = torch.mean(en_cri(poutput))
                print('[DEBUG] itr {} ae entropy before: {}'.format(i, en_loss))

            #acc_loss = criterion(output, target)
            # print('[DEBUG] itr {} ae acc_loss: {}'.format(i, acc_loss))
            #loss = acc_loss
            loss = -en_loss
            loss.backward()
            grad = delta.grad.detach()

            #idx_update = torch.ones(target.shape, dtype=torch.bool)
            grad_sign = sign(grad)
            #delta.data[idx_update] = (delta + (eps / 4) * grad_sign)[idx_update]
            delta.data = delta + (eps / 4) * grad_sign
            delta.data = clamp(torch.add(x, delta.data), 0, 1) - x
            delta.data = clamp(delta.data, -eps, eps)
            delta.grad.zero_()
            ae_x = clamp(torch.add(x, delta), 0, 1)
            output = model(ae_x)
            for pmodel in pmodels:
                poutput = pmodel(ae_x).view(len(x), -1)
                en_loss = torch.mean(en_cri(poutput))
                print('[DEBUG] itr {} ae entropy after: {}'.format(i, en_loss))

    return delta.detach()


def ae_training_tgt(model, x, y, criterion, mean, std, attack_iters=10, eps=0.0392, rs=True):
    delta = torch.zeros_like(x).cuda()
    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True
    for i in range(attack_iters):
        ae_x = clamp(x + delta, 0, 1)
        output = model(ae_x)

        tgt = (torch.ones(len(ae_x), dtype=torch.int64) * 150).cuda()
        loss = criterion(output, tgt)
        loss.backward()
        grad = delta.grad.detach()

        delta_out_class = torch.argmax(output, dim=-1)
        print('[DEBUG] iteration {} delta_out_class: {}'.format(i, delta_out_class))

        idx_update = torch.ones(y.shape, dtype=torch.bool)
        grad_sign = sign(grad)
        delta.data[idx_update] = (delta + (eps / 4) * grad_sign)[idx_update]
        delta.data = clamp(x + delta.data, 0, 1) - x
        delta.data = clamp(delta.data, -eps, eps)
        delta.grad.zero_()


    return delta.detach()


def ae_pgd_tgt(model, x, y, criterion, attack_iters=10, eps=0.0392, rs=True):
    x_adv = x.detach().clone().requires_grad_(True).cuda()

    for i in range(attack_iters):
        pred = model(x_adv)
        tgt = (torch.ones(len(x_adv), dtype=torch.int64) * 150).cuda()
        loss = criterion(pred, tgt)
        loss.backward()

        grad = x_adv.grad.data.detach()
        grad_unit = grad / grad.norm(p=eps, dim=(2, 3), keepdim=True)
        x_adv = x_adv + grad_unit * (eps / 4)
        x_adv = clamp(x_adv, 0, 1)
        model.zero_grad()
        delta_out_class = torch.argmax(pred, dim=-1)
        print('[DEBUG] iteration {} delta_out_class: {}'.format(i, delta_out_class))

    return x_adv.detach().cuda()


def known_uap_train(data_loader,
                    model,
                    arch,
                    criterion,
                    optimizer,
                    num_iterations,
                    split_layers,
                    uaps,
                    alpha=0.1,
                    use_cuda=True):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    print('[DEBUG] dataloader length: {}'.format(len(data_loader)))

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            # measure data loading time
            data_time.update(time.time() - end)

            # select a uap
            indices = torch.randperm(len(uaps))[:len(input)]
            uap = uaps[indices]
            pert_input = input
            for i, uap_i in enumerate(uap):
                if torch.randint(low=0, high=2, size=(1,1)):
                    pert_input[i] = input[i] + uap_i

            if use_cuda:
                input = input.cuda()
                target = target.cuda()
                pert_input = pert_input.cuda()

             # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                # cce loss only
                poutput = model((pert_input).float())
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            adv_prec1, adv_prec5 = accuracy(poutput.data, target_, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/1563]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      .format(
                    num_batch, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5,
                    adv_top1=adv_top1, adv_top5=adv_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
              'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
              .format(
               iteration, num_iterations, batch_time=batch_time,
               adv_top1=adv_top1, adv_top5=adv_top5, data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model


def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign


def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def standardize_delta(delta, mean, std):
    return (delta - mean) / std


def destandardize_delta(delta, mean, std):
    return (delta * std + mean)


def metrics_evaluate(data_loader, target_model, perturbed_model, targeted, target_class, log=None, use_cuda=True):
    # switch to evaluate mode
    target_model.eval()
    perturbed_model.eval()
    perturbed_model.module.generator.eval()
    perturbed_model.module.target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            pert_output = perturbed_model(input)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(pert_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        pert_out_class = torch.argmax(pert_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == pert_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == pert_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = perturbed_model(input[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg != 0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(pert_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), pert_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = pert_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    if log:
        print_log('\n\t#######################', log)
        print_log('\tClean model accuracy: {:.3f}'.format(clean_acc.avg), log)
        print_log('\tPerturbed model accuracy: {:.3f}'.format(perturbed_acc.avg), log)
        print_log('\tAbsolute Accuracy Drop: {:.3f}'.format(aad_source), log)
        print_log('\tRelative Accuracy Drop: {:.3f}'.format(rad_source), log)
        print_log('\tAttack Success Rate: {:.3f}'.format(100-attack_success_rate.avg), log)
        print_log('\tFooling Ratio: {:.3f}'.format(fooling_ratio), log)
        if targeted:
            print_log('\tAll --> Target Class {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate.avg), log)
            print_log('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate_filtered.avg), log)



def metrics_evaluate_test(data_loader, target_model, uap, targeted, target_class, mask=None, log=None, use_cuda=True):
    # switch to evaluate mode
    target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            if mask is None:
                adv_x = (input + uap).float()
            else:
                adv_x = torch.mul((1 - mask), input) + torch.mul(mask, uap).float()
            attack_output = target_model(adv_x)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(attack_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        uap_out_class = torch.argmax(attack_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == uap_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == uap_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = target_model(adv_x[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg != 0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(attack_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), attack_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = attack_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    print('\n\t#######################')
    print('\tClean model accuracy: {:.3f}'.format(clean_acc.avg))
    print('\tPerturbed model accuracy: {:.3f}'.format(perturbed_acc.avg))
    print('\tAbsolute Accuracy Drop: {:.3f}'.format(aad_source))
    print('\tRelative Accuracy Drop: {:.3f}'.format(rad_source))
    print('\tAttack Success Rate: {:.3f}'.format(100-attack_success_rate.avg))
    print('\tFooling Ratio: {:.3f}'.format(fooling_ratio))
    if targeted:
        print('\tAll --> Target Class {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate.avg))
        print('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate_filtered.avg))



def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def solve_causal(data_loader, filter_model, uap, filter_arch, targeted, target_class, num_sample, split_layer=43, causal_type='logit', log=None, use_cuda=True):
    '''
    perform causality analysis on the dense layer before logit layer
    Args:
        data_loader: loader that loads original images with uap
        filter_model:
        uap:
        filter_arch:
        target_class:
        num_sample: number of samples to use for causality analysis
        causal_type:
            - logit: analyze ACE of dense layer neuron on logits
            - act: analyze ACE of uap on dense layer
        log:
        use_cuda:

    Returns:

    '''
    #split the model
    model1, model2 = split_model(filter_model, filter_arch, split_layer=split_layer)

    # switch to evaluate mode
    model1.eval()
    model2.eval()
    #filter_model.eval()
    out = []
    if causal_type == 'logit':
        if not targeted:
            return None
        total_num_samples = 0
        out = []
        do_predict_avg = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                if uap != None:
                    uap = uap.cuda().float()
            if uap != None:
                input = input + uap

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                ori_output = model2(dense_output)

                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                #ori_output_ = filter_model(input + uap)
                do_predict_neu = []
                do_predict = []
                #do convention for each neuron
                for i in range(0, len(dense_hidden_[0])):
                    hidden_do = np.zeros(shape=dense_hidden_[:, i].shape)
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = torch.from_numpy(hidden_do)
                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    if use_cuda:
                        dense_output_ = dense_output_.cuda()
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                do_predict_neu = np.array(do_predict_neu)
                do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.mean(np.array(do_predict_neu), axis=1)  #4096x10

            do_predict_avg.append(do_predict) #batchx4096x11
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0) #4096x10
        # insert neuron index
        idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        do_predict_avg = np.c_[idx, do_predict_avg]
        out = do_predict_avg[:, [0, (target_class + 1)]]
    elif causal_type == 'act':
        total_num_samples = 0
        dense_avg = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                if uap != None:
                    uap = uap.cuda().float()
            if uap != None:
                input = input + uap

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                # ori_output = model2(dense_output)
                dense_this = torch.reshape(dense_output, (dense_output.shape[0], -1)).cpu().detach().numpy()
                dense_this = np.mean(dense_this, axis=0)  # 4096
            dense_avg.append(dense_this)  # batchx4096
            total_num_samples += len(gt)
        # average of all baches
        dense_avg = np.mean(np.array(dense_avg), axis=0)  # 4096
        # insert neuron index
        idx = np.arange(0, len(dense_avg), 1, dtype=int)
        dense_avg = np.c_[idx, dense_avg]
        out = dense_avg
        #print('shape of dense_avg {}'.format(dense_avg.shape))
    if causal_type == 'slogit':
        if not targeted:
            return None
        total_num_samples = 0
        do_predict_avg = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda().float()

            # compute output
            with torch.no_grad():
                dense_output = model1(input + uap)
                ori_output = model2(dense_output)
                ori_output_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
                filter_mask = (ori_output_class == target_class)
                dense_output = dense_output.cpu().numpy()[filter_mask]

                if len(dense_output) == 0:
                    continue
                dense_output = torch.from_numpy(dense_output)
                do_predict_neu = []
                #do convention for each neuron
                for i in range(0, len(dense_output[0])):
                    hidden_do = np.zeros(shape=dense_output[:, i].shape)
                    dense_output_ = torch.clone(dense_output)
                    dense_output_[:, i] = torch.from_numpy(hidden_do)
                    if use_cuda:
                        dense_output_ = dense_output_.cuda()
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                do_predict_neu = np.array(do_predict_neu)
                do_predict_neu = np.abs(ori_output.cpu().detach().numpy()[filter_mask] - do_predict_neu)
                do_predict = np.mean(np.array(do_predict_neu), axis=1)  #4096x10
                #do_predict = do_predict_neu[:,target_class]

            do_predict_avg.append(do_predict) #batchx4096x11
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0) #4096x10
        # insert neuron index
        idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        do_predict_avg = np.c_[idx, do_predict_avg]
        out = do_predict_avg[:, [0, (target_class + 1)]]

    elif causal_type == 'sact':
        if not targeted:
            return None
        total_num_samples = 0
        dense_avg = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda().float()

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                pert_dense_output = model1(input + uap)
                pert_output = model2(pert_dense_output)

                pert_output_class = torch.argmax(pert_output, dim=-1).cpu().numpy()
                filter_mask = (pert_output_class == target_class)

                dense_output = dense_output.cpu().detach().numpy()[filter_mask]
                pert_dense_output = pert_dense_output.cpu().detach().numpy()[filter_mask]

                dense_this = np.abs(dense_output - pert_dense_output)# 4096
                dense_this = np.mean(dense_this, axis=0)  # 4096

            dense_avg.append(dense_this)  # batchx4096
            total_num_samples += len(gt)
        # average of all baches
        dense_avg = np.mean(np.array(dense_avg), axis=0)  # 4096
        # insert neuron index
        idx = np.arange(0, len(dense_avg), 1, dtype=int)
        dense_avg = np.c_[idx, dense_avg]
        out = dense_avg

    elif causal_type == 'be_act':   # sample from target class activation
        if not targeted:
            return None
        total_num_samples = 0
        dense_avg = []
        num_target_sample = 0
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda().float()

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                dense_hidden_ = dense_hidden_.cpu().detach().numpy()
                dense_hidden_ = dense_hidden_[(gt.cpu().detach().numpy() == target_class), :]
                num_target_sample += len(dense_hidden_)
                dense_this = np.sum(dense_hidden_, axis=0)  # 4096

            dense_avg.append(dense_this)  # batchx4096
            total_num_samples += len(gt)
        # average of all baches
        dense_avg = np.sum(np.array(dense_avg), axis=0) / num_target_sample  # 4096
        # insert neuron index
        idx = np.arange(0, len(dense_avg), 1, dtype=int)
        dense_avg = np.c_[idx, dense_avg]
        out = dense_avg

    elif causal_type == 'uap_act':
        if not targeted:
            return None
        if use_cuda:
            uap = uap.cuda().float()
            # compute output
        with torch.no_grad():
            dense_output = model1(uap)
            dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
            #uap_output = filter_model(uap)
            #uap_output_class = torch.argmax(uap_output, dim=-1).cpu().numpy()
            dense_this = dense_hidden_.cpu().detach().numpy().transpose() #4096

        # insert neuron index
        idx = np.arange(0, len(dense_this), 1, dtype=int)
        dense_this = np.c_[idx, dense_this]
        out = dense_this

    elif causal_type == 'inact':
        # find inactive neurons; untargeted attack
        total_num_samples = 0
        dense_avg = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                # ori_output = model2(dense_output)
                dense_this = dense_output.cpu().detach().numpy() # 32x4096
                dense_this = np.mean(dense_this, axis=0)  # 4096
            dense_avg.append(dense_this)  # batchx4096
            total_num_samples += len(gt)

        # average of all baches
        dense_avg = np.mean(np.array(dense_avg), axis=0)  # 4096
        # invert for ranking later
        #dense_avg = 1 - dense_avg / np.max(dense_avg)
        my_max = np.max(dense_avg)

        # insert neuron index
        idx = np.arange(0, len(dense_avg), 1, dtype=int)
        dense_avg = np.c_[idx, dense_avg]

        temp = dense_avg
        ind = np.argsort(temp[:, 1])#[::-1]
        dense_avg = temp[ind]
        for i in range (len(dense_avg)):
            if dense_avg[i][1] > 0.1 * my_max:
                break
        out = dense_avg[:i]

    return out


def solve_causal_single(data_loader, filter_model, uap, filter_arch, targeted, target_class, num_sample, split_layer=43, causal_type='logit', log=None, use_cuda=True):
    '''
    perform causality analysis on the dense layer before logit layer
    Args:
        data_loader: loader that loads original images with uap
        filter_model:
        uap:
        filter_arch:
        target_class:
        num_sample: number of samples to use for causality analysis
        causal_type:
            - logit: analyze ACE of dense layer neuron on logits
            - act: analyze ACE of uap on dense layer
        log:
        use_cuda:

    Returns:

    '''
    #split the model
    model1, model2 = split_model(filter_model, filter_arch, split_layer=split_layer)

    #test


    # switch to evaluate mode
    model1.eval()
    model2.eval()
    #filter_model.eval()
    out = []
    if causal_type == 'logit':
        if not targeted:
            return None
        total_num_samples = 0
        out = []
        do_predict_avg = []
        outputs = []
        clean_outputs = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break

            ori_input = input
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                ori_input = ori_input.cuda()
                if uap != None:
                    uap = uap.cuda().float()

            if uap != None:
                input = input + uap

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                ori_output = model2(dense_output)
                clean_output = torch.argmax(filter_model(ori_input), dim=-1).cpu().numpy()
                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                ori_output_ = filter_model(input)
                ori_out_class = torch.argmax(ori_output_, dim=-1).cpu().numpy()
                outputs = outputs + list(ori_out_class)
                clean_outputs = clean_outputs + list(clean_output)
                do_predict_neu = []
                do_predict = []
                #do convention for each neuron
                for i in range(0, len(dense_hidden_[0])):
                    hidden_do = np.zeros(shape=dense_hidden_[:, i].shape)
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = torch.from_numpy(hidden_do)
                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    if use_cuda:
                        dense_output_ = dense_output_.cuda()
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                do_predict_neu = np.array(do_predict_neu)
                do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.array(do_predict_neu)

            do_predict = list(np.transpose(do_predict, (1, 0, 2)))
            do_predict_avg = do_predict_avg + do_predict
            total_num_samples += len(gt)
        # average of all baches
        out = np.array(do_predict_avg) #4096x10
        # insert neuron index
        #idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        #do_predict_avg = np.c_[idx, do_predict_avg]
        #out = do_predict_avg[:, [0, (target_class + 1)]]
    elif causal_type == 'act':
        total_num_samples = 0
        dense_avg = []
        outputs = []
        clean_outputs = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break

            ori_input = input
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                ori_input = ori_input.cuda()
                if uap != None:
                    uap = uap.cuda().float()
            if uap != None:
                input = input + uap

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                ori_output = model2(dense_output)
                #test
                #test_output = filter_model(input)
                clean_output = torch.argmax(filter_model(ori_input), dim=-1).cpu().numpy()
                ori_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
                outputs = outputs + list(ori_out_class)
                clean_outputs = clean_outputs + list(clean_output)
                dense_this = torch.reshape(dense_output, (dense_output.shape[0], -1)).cpu().detach().numpy()# 4096
            dense_avg = dense_avg + list(dense_this)  # batchx4096
            total_num_samples += len(gt)
        # average of all baches
        dense_avg = np.array(dense_avg)# 4096
        #print('shape of dense_avg {}'.format(dense_avg.shape))
        # insert neuron index
        out = dense_avg
    return out, outputs, clean_outputs


def solve_causal_ae(data_loader, filter_model, aes, filter_arch, targeted, target_class, num_sample, split_layer=43, causal_type='logit', log=None, use_cuda=True):
    '''
    perform causality analysis on the dense layer before logit layer
    Args:
        data_loader: loader that loads original images with uap
        filter_model:
        aes:
        filter_arch:
        target_class:
        num_sample: number of samples to use for causality analysis
        causal_type:
            - logit: analyze ACE of dense layer neuron on logits
            - act: analyze ACE of uap on dense layer
        log:
        use_cuda:

    Returns:

    '''
    #split the model
    model1, model2 = split_model(filter_model, filter_arch, split_layer=split_layer)

    #test


    # switch to evaluate mode
    model1.eval()
    model2.eval()
    #filter_model.eval()
    out = []
    if causal_type == 'act':
        total_num_samples = 0
        dense_avg = []
        outputs = []
        clean_outputs = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break

            ori_input = input
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                ori_input = ori_input.cuda()
                if aes != None:
                    aes = aes.cuda().float()
            if aes != None:
                input = input + aes[total_num_samples]

            # compute output
            with torch.no_grad():
                dense_output = model1(input)
                ori_output = model2(dense_output)
                #test
                #test_output = filter_model(input)
                clean_output = torch.argmax(filter_model(ori_input), dim=-1).cpu().numpy()
                ori_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
                outputs = outputs + list(ori_out_class)
                clean_outputs = clean_outputs + list(clean_output)
                dense_this = torch.reshape(dense_output, (dense_output.shape[0], -1)).cpu().detach().numpy()# 4096
            dense_avg = dense_avg + list(dense_this)  # batchx4096
            total_num_samples += len(gt)
        # average of all baches
        dense_avg = np.array(dense_avg)# 4096
        #print('shape of dense_avg {}'.format(dense_avg.shape))
        # insert neuron index
        out = dense_avg
    return out, outputs, clean_outputs


def solve_causal_single_both(data_loader, filter_model, uap, filter_arch, targeted, target_class, num_sample, split_layer=43, causal_type='logit', log=None, use_cuda=True):
    '''
    perform causality analysis on the dense layer before logit layer
    Args:
        data_loader: loader that loads original images with uap
        filter_model:
        uap:
        filter_arch:
        target_class:
        num_sample: number of samples to use for causality analysis
        causal_type:
            - logit: analyze ACE of dense layer neuron on logits
            - act: analyze ACE of uap on dense layer
        log:
        use_cuda:

    Returns:

    '''
    #split the model
    if uap == None:
        return

    model1, model2 = split_model(filter_model, filter_arch, split_layer=split_layer)

    # switch to evaluate mode
    model1.eval()
    model2.eval()
    #filter_model.eval()

    total_num_samples = 0
    pert_dense_avg = []
    pert_outputs = []
    clean_dense_avg = []
    clean_outputs = []
    for input, gt in data_loader:
        if total_num_samples >= num_sample:
            break

        ori_input = input
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()
            ori_input = ori_input.cuda()
            if uap != None:
                uap = uap.cuda().float()

        input = input + uap

        # compute output
        with torch.no_grad():
            perturb_dense = model1(input)
            perturb_output = model2(perturb_dense)
            pert_out_class = torch.argmax(perturb_output, dim=-1).cpu().numpy()
            pert_outputs = pert_outputs + list(pert_out_class)
            pert_dense_this = torch.reshape(perturb_dense, (perturb_dense.shape[0], -1)).cpu().detach().numpy()

            clean_dense = model1(ori_input)
            clean_output = model2(clean_dense)
            clean_output_classs = torch.argmax(clean_output, dim=-1).cpu().numpy()
            clean_outputs = clean_outputs + list(clean_output_classs)
            clean_dense_this = torch.reshape(clean_dense, (clean_dense.shape[0], -1)).cpu().detach().numpy()

        pert_dense_avg = pert_dense_avg + list(pert_dense_this)
        clean_dense_avg = clean_dense_avg + list(clean_dense_this)

        total_num_samples += len(gt)
    # average of all baches
    pert_dense_avg = np.array(pert_dense_avg)
    clean_dense_avg = np.array(clean_dense_avg)

    return clean_dense_avg, pert_dense_avg


def solve_causal_uap(filter_model, uap, filter_arch, split_layer=43, causal_type='logit', use_cuda=True):

    #split the model
    model1, model2 = split_model(filter_model, filter_arch, split_layer=split_layer)

    # switch to evaluate mode
    model1.eval()
    model2.eval()
    #filter_model.eval()
    out = []
    if causal_type == 'act':
        dense_avg = []
        outputs = []

        if use_cuda:
            uap = uap.cuda().float()
            input = uap

        # compute output
        with torch.no_grad():
            dense_output = model1(input)
            ori_output = model2(dense_output)
            #test
            #test_output = filter_model(input)
            ori_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
            outputs = outputs + list(ori_out_class)
            dense_this = torch.reshape(dense_output, (dense_output.shape[0], -1)).cpu().detach().numpy()# 4096
        dense_avg = dense_avg + list(dense_this)  # batchx4096

        # insert neuron index
        out = dense_avg
    return out, outputs


def my_test_uap(data_loader, filter_model, uap, target_class, num_sample, use_cuda=True):
    model = filter_model
    # switch to evaluate mode
    model.eval()

    total_num_samples = 0
    num_correct = 0
    num_fool = 0
    num_target = 0
    for input, gt in data_loader:
        if total_num_samples >= num_sample:
            break
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()
            if uap != None:
                uap = uap.cuda().float()
        if uap != None:
            pertub_input = input + uap

        # compute output
        with torch.no_grad():
            ori_output_ = model(input)
            ori_out_class = torch.argmax(ori_output_, dim=-1).cpu().numpy()
            pert_output = model(pertub_input)
            pert_out_class = torch.argmax(pert_output, dim=-1).cpu().numpy()
            #print('[DEBUG] ori_class {} pert class {} expected class {}'.format(ori_out_class, pert_out_class, gt.cpu().numpy()))
            num_correct += np.sum(ori_out_class == gt.cpu().numpy())
            num_fool += np.sum(
                            np.logical_and((pert_out_class != gt.cpu().numpy()), (ori_out_class == gt.cpu().numpy()))
                        )
            num_target += np.sum(pert_out_class == target_class)
        total_num_samples += len(gt)

    out = num_correct / total_num_samples * 100.
    fr = num_fool / num_correct * 100.
    asr = num_target / total_num_samples * 100.
    return num_correct, out, num_fool, fr, total_num_samples, asr


def my_test(data_loader, filter_model, uap, target_class, num_sample, split_layer=43, use_cuda=True):
    model = filter_model
    # switch to evaluate mode
    model.eval()

    total_num_samples = 0
    num_correct = 0
    for input, gt in data_loader:
        if total_num_samples >= num_sample:
            break
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()
            if uap != None:
                uap = uap.cuda().float()
        if uap != None:
            input = input + uap

        # compute output
        with torch.no_grad():
            ori_output_ = model(input)
            ori_out_class = torch.argmax(ori_output_, dim=-1).cpu().numpy()
            num_correct += np.sum(ori_out_class == gt.cpu().numpy())
        total_num_samples += len(gt)

    out = num_correct / total_num_samples * 100.

    return num_correct, out, total_num_samples


def solve_input_attribution(data_loader, model, uap, targeted, target_class, num_sample, causal_type='logit', use_cuda=True):
    # switch to evaluate mode
    model.eval()

    out = []
    if causal_type == 'logit':
        if not targeted:
            return None
        total_num_samples = 0
        do_predict_avg = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                if uap != None:
                    uap = uap.cuda().float()
            if uap != None:
                test_input = input + uap
            else:
                test_input = input

            # compute output
            with torch.no_grad():
                uap_output = model(test_input)

                #intervention
                uap_input_flat = torch.clone(torch.reshape(test_input, (test_input.shape[0], -1)))

                do_predict = []
                #do convention for each neuron
                for i in range(0, len(uap_input_flat[0])):
                    input_do = np.zeros(shape=uap_input_flat[:, i].shape)
                    dense_output_ = torch.clone(uap_input_flat)
                    dense_output_[:, i] = torch.from_numpy(input_do)
                    dense_output_ = torch.reshape(dense_output_, input.shape)
                    if use_cuda:
                        dense_output_ = dense_output_.cuda()
                    output_do = model(dense_output_).cpu().detach().numpy()
                    do_predict.append(output_do)

                do_predict_neu = np.array(do_predict)
                do_predict_neu = np.abs(uap_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.mean(np.array(do_predict_neu), axis=1)

            do_predict_avg.append(do_predict)
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)
        # insert neuron index
        idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        do_predict_avg = np.c_[idx, do_predict_avg]
        out = do_predict_avg[:, [0, (target_class + 1)]]

    return out


def solve_input_attribution_single(data_loader, model, uap, targeted, target_class, num_sample, causal_type='logit', use_cuda=True):
    # switch to evaluate mode
    model.eval()

    out = []
    if causal_type == 'logit':
        if not targeted:
            return None
        total_num_samples = 0
        do_predict_avg = []
        for input, gt in data_loader:
            if total_num_samples >= num_sample:
                break
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                if uap != None:
                    uap = uap.cuda().float()
            if uap != None:
                test_input = input + uap
            else:
                test_input = input

            # compute output
            with torch.no_grad():
                uap_output = model(test_input)

                #intervention
                uap_input_flat = torch.clone(torch.reshape(test_input, (test_input.shape[0], -1)))

                do_predict = []
                #do convention for each neuron
                for i in range(0, len(uap_input_flat[0])):
                    input_do = np.zeros(shape=uap_input_flat[:, i].shape)
                    dense_output_ = torch.clone(uap_input_flat)
                    dense_output_[:, i] = torch.from_numpy(input_do)
                    dense_output_ = torch.reshape(dense_output_, input.shape)
                    if use_cuda:
                        dense_output_ = dense_output_.cuda()
                    output_do = model(dense_output_).cpu().detach().numpy()
                    do_predict.append(output_do)

                do_predict_neu = np.array(do_predict)
                do_predict_neu = np.abs(uap_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.array(do_predict_neu)

            do_predict_avg.append(do_predict)
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)
        # insert neuron index
        #idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        #do_predict_avg = np.c_[idx, do_predict_avg]
        #out = do_predict_avg[:, :, :target_class]
        out = np.transpose(out, (1, 0, 2))

    return out



def solve_activation(data_loader, filter_model, uap, filter_arch, target_class, num_sample, log=None, use_cuda=True):
    '''
    find most active neurons
    Args:
        data_loader:
        filter_model:
        uap:
        filter_arch:
        target_class:
        num_sample:
        log:
        use_cuda:

    Returns:

    '''
    #split the model
    model1, model2 = split_model(filter_model, filter_arch)

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    total_num_samples = 0
    dense_avg = []
    for input, gt in data_loader:
        if total_num_samples >= num_sample:
            break
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()
            uap = uap.cuda().float()

        # compute output
        with torch.no_grad():
            dense_output = model1(input + uap)
            #ori_output = model2(dense_output)
            dense_this = np.mean(dense_output.cpu().detach().numpy(), axis=0) #4096

        dense_avg.append(dense_this) #batchx4096
        total_num_samples += len(gt)
    # average of all baches
    dense_avg = np.mean(np.array(dense_avg), axis=0) #4096x10
    # insert neuron index
    idx = np.arange(0, len(dense_avg), 1, dtype=int)
    dense_avg = np.c_[idx, dense_avg]

    return dense_avg


def eval_uap(test_data_loader, target_model, uap, target_class, log=None, use_cuda=True, targeted=True):


    # switch to evaluate mode
    target_model.eval()
    if targeted:
        '''
        total_num_samples = 0
        num_attack_success = 0
        for input, gt in train_data_loader:
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda()
    
            # compute output
            with torch.no_grad():
                attack_output = target_model(input + uap)
                ori_output = target_model(input)
    
            # Calculating Fooling Ratio params
            #clean_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
            pert_out_class = torch.argmax(attack_output, dim=-1).cpu().numpy()
    
            num_attack_success += np.sum(pert_out_class == target_class)
    
            total_num_samples += len(gt)
        train_sr = num_attack_success / total_num_samples * 100
        '''
        total_num_samples = 0
        num_attack_success = 0
        num_non_t_succ = 0
        clean_correctly_classified = 0
        #exclude samples from target class
        _num_attack_success = 0
        _num_non_t_succ = 0
        _total_num_samples = 0
        for input, gt in test_data_loader:
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda().float()

            # compute output
            with torch.no_grad():
                attack_output = target_model(input + uap)
                ori_output = target_model(input)

            # Calculating Fooling Ratio params
            clean_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
            pert_out_class = torch.argmax(attack_output, dim=-1).cpu().numpy()

            clean_correctly_classified += np.sum(clean_out_class == gt.cpu().numpy())
            num_attack_success += np.sum(pert_out_class == target_class)
            num_non_t_succ += np.sum(pert_out_class != gt.cpu().numpy())

            #exclude samples from target class
            #'''
            non_target_class_mask = (gt.cpu().numpy() != target_class)
            if np.sum(non_target_class_mask) > 0:
                _num_attack_success += np.sum((pert_out_class == target_class) * non_target_class_mask)
                _num_non_t_succ += np.sum((pert_out_class != gt.cpu().numpy()) * non_target_class_mask)
                _total_num_samples += np.sum(gt.cpu().numpy() != target_class)
            #'''
            total_num_samples += len(gt)
        test_sr = num_attack_success / total_num_samples * 100
        clean_test_acc = clean_correctly_classified / total_num_samples * 100
        nt_sr = num_non_t_succ / total_num_samples * 100

        _test_sr = _num_attack_success / _total_num_samples * 100
        _nt_sr = _num_non_t_succ / _total_num_samples * 100
    else:
        total_num_samples = 0
        num_non_t_succ = 0
        clean_correctly_classified = 0
        # exclude samples from target class
        _num_attack_success = 0
        _num_non_t_succ = 0
        _total_num_samples = 0
        for input, gt in test_data_loader:
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda().float()

            # compute output
            with torch.no_grad():
                attack_output = target_model(input + uap)
                ori_output = target_model(input)

            # Calculating Fooling Ratio params
            clean_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
            pert_out_class = torch.argmax(attack_output, dim=-1).cpu().numpy()

            clean_correctly_classified += np.sum(clean_out_class == gt.cpu().numpy())
            num_non_t_succ += np.sum(pert_out_class != gt.cpu().numpy())

            total_num_samples += len(gt)
        test_sr = 0
        clean_test_acc = clean_correctly_classified / total_num_samples * 100
        nt_sr = num_non_t_succ / total_num_samples * 100

        _test_sr = 0
        _nt_sr = nt_sr
    return test_sr, nt_sr, clean_test_acc, _test_sr, _nt_sr


def eval_uap_model(test_data_loader, target_model, pert_model, target_class, log=None, use_cuda=True, targeted=True):
    # switch to evaluate mode
    pert_model.eval()
    target_model.eval()
    if targeted:
        total_num_samples = 0
        num_attack_success = 0
        num_non_t_succ = 0
        clean_correctly_classified = 0
        # exclude samples from target class
        _num_attack_success = 0
        _num_non_t_succ = 0
        _total_num_samples = 0
        for input, gt in test_data_loader:
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda().float()

            # compute output
            with torch.no_grad():
                attack_output = pert_model(input)
                ori_output = target_model(input)

            # Calculating Fooling Ratio params
            clean_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
            pert_out_class = torch.argmax(attack_output, dim=-1).cpu().numpy()

            clean_correctly_classified += np.sum(clean_out_class == gt.cpu().numpy())
            num_attack_success += np.sum(pert_out_class == target_class)
            num_non_t_succ += np.sum(pert_out_class != gt.cpu().numpy())

            # exclude samples from target class
            # '''
            non_target_class_mask = (gt.cpu().numpy() != target_class)
            if np.sum(non_target_class_mask) > 0:
                _num_attack_success += np.sum((pert_out_class == target_class) * non_target_class_mask)
                _num_non_t_succ += np.sum((pert_out_class != gt.cpu().numpy()) * non_target_class_mask)
                _total_num_samples += np.sum(gt.cpu().numpy() != target_class)
            # '''
            total_num_samples += len(gt)
        test_sr = num_attack_success / total_num_samples * 100
        clean_test_acc = clean_correctly_classified / total_num_samples * 100
        nt_sr = num_non_t_succ / total_num_samples * 100

        _test_sr = _num_attack_success / _total_num_samples * 100
        _nt_sr = _num_non_t_succ / _total_num_samples * 100
    else:
        total_num_samples = 0
        num_non_t_succ = 0
        clean_correctly_classified = 0
        # exclude samples from target class
        _num_attack_success = 0
        _num_non_t_succ = 0
        _total_num_samples = 0
        for input, gt in test_data_loader:
            if use_cuda:
                gt = gt.cuda()
                input = input.cuda()
                uap = uap.cuda().float()

            # compute output
            with torch.no_grad():
                attack_output = target_model(input + uap)
                ori_output = target_model(input)

            # Calculating Fooling Ratio params
            clean_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
            pert_out_class = torch.argmax(attack_output, dim=-1).cpu().numpy()

            clean_correctly_classified += np.sum(clean_out_class == gt.cpu().numpy())
            num_non_t_succ += np.sum(pert_out_class != gt.cpu().numpy())

            total_num_samples += len(gt)
        test_sr = 0
        clean_test_acc = clean_correctly_classified / total_num_samples * 100
        nt_sr = num_non_t_succ / total_num_samples * 100

        _test_sr = 0
        _nt_sr = nt_sr
    return test_sr, nt_sr, clean_test_acc, _test_sr, _nt_sr


def replace_model(ori_model, model_name, replace_layer=38):
    '''
    reconstruct given model with costumized pooling layer
    '''
    if model_name == 'vgg19':
        if replace_layer == 38:
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:37]
            module3 = layers[38:]
            out_model = nn.Sequential(*[*module1, MyAvgPool2D(output_size=(7, 7)), Flatten(), *module3])
        elif replace_layer <= 37:
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:replace_layer-1]
            module2 = layers[replace_layer:38]
            module3 = layers[38:]
            out_model = nn.Sequential(*[*module1, MyMaxPool2D(kernel_size=2, stride=2, replace=True),
                                        *module2, Flatten(), *module3])
        elif replace_layer == 43:
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:38]
            module2 = layers[38:replace_layer]
            module3 = layers[replace_layer:]
            out_model = nn.Sequential(*[*module1, Flatten(), *module2, EntropyWeight(), *module3])
        else:
            return ori_model
    else:
        return ori_model

    return out_model


def split_model(ori_model, model_name, split_layer=43, flat=False):
    '''
    split given model from the dense layer before logits
    Args:
        ori_model:
        model_name: model name
    Returns:
        splitted models
    '''
    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer < 9:
            modules = list(ori_model.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:9]
            module3 = modules[9:]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[:9]
            module2 = modules[9]

            model_1st = nn.Sequential(*module1, Flatten())
            model_2nd = nn.Sequential(*[module2])

        else:
            return None, None
    elif model_name == 'googlenet':
        if split_layer < 17:
            modules = list(ori_model.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:17]
            module3 = modules[17:]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 17:
            modules = list(ori_model.children())
            module1 = modules[:17]
            module2 = modules[17:]

            model_1st = nn.Sequential(*module1, Flatten())
            model_2nd = nn.Sequential(*module2)

        elif split_layer == 19:  # googlenet caffe
            modules = list(ori_model.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:22]
            module3 = modules[22:]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 22: # googlenet caffe
            modules = list(ori_model.children())
            module1 = modules[:22]
            module2 = modules[22:]

            model_1st = nn.Sequential(*module1, Flatten())
            model_2nd = nn.Sequential(*module2)
        else:
            return None, None
    elif model_name == 'vgg19':
        if flat:
            layers = list(ori_model.children())
            module1 = layers[:split_layer]
            module2 = layers[split_layer:]
            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*module2)
        else:
            if split_layer < 38:
                modules = list(ori_model.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:split_layer]
                module2 = layers[split_layer:38]
                module3 = layers[38:]
                model_1st = nn.Sequential(*module1)
                model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])
            else:
                modules = list(ori_model.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:38]
                moduel2 = layers[38:split_layer]
                module3 = layers[split_layer:]
                model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
                model_2nd = nn.Sequential(*module3)
    elif model_name == 'alexnet':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[0]
            module2 = [modules[1]]
            module_ = list(modules[2])
            module3 = module_[:5]
            module4 = module_[5:]

            model_1st = nn.Sequential(*[*module1, Flatten(), *module2, *module3])
            model_2nd = nn.Sequential(*module4)
    elif model_name == 'shufflenetv2':
        if split_layer == 6:
            modules = list(ori_model.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = modules[1:6]
            module2 = [sub_modules[0]]
            module3 = [sub_modules[1]]

            model_1st = nn.Sequential(*[*module0, *module1, Avgpool2d_n(poolsize=7), Flatten(), *module2])
            model_2nd = nn.Sequential(*module3)
        elif split_layer == 1:
            modules = list(ori_model.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = modules[2:6]
            module3 = sub_modules

            model_1st = nn.Sequential(*[*module0, *module1])
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'mobilenet':
        if split_layer == 3:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = [modules[3]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(ori_model.children())
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = [modules[2]]
            module3 = [modules[3]]
            model_1st = nn.Sequential(*module0)
            model_2nd = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'wideresnet':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[3:7]
            module3 = [modules[-1]]

            model_1st = nn.Sequential(*[*module1, *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(ori_model.children())
            module1 = modules[:2]

            module2 = modules[3:7]
            module3 = [modules[-1]]
            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    else:
        return None, None

    #summary(ori_model, (3, 224, 224))
    #summary(model_1st, (3, 224, 224))

    return model_1st, model_2nd


def split_perturbed_model(ori_model, model_name, split_layer=43, flat=False):
    modules = list(ori_model.module.children())
    generator = modules[0]
    target_network = modules[1]

    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer < 9:
            modules = list(target_network.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:9]
            module3 = modules[9:]

            model_1st = nn.Sequential(*[*[generator], *module1])
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 9:
            modules = list(target_network.children())
            module1 = modules[:9]
            module2 = modules[9]

            model_1st = nn.Sequential(*[generator], *module1, Flatten())
            model_2nd = nn.Sequential(*[module2])

        else:
            return None, None
    elif model_name == 'googlenet':
        if split_layer < 17:
            modules = list(target_network.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:17]
            module3 = modules[17:]

            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 17:
            modules = list(target_network.children())
            module1 = modules[:17]
            module2 = modules[17:]

            model_1st = nn.Sequential(*[generator], *module1, Flatten())
            model_2nd = nn.Sequential(*module2)

        elif split_layer == 19:  # googlenet caffe
            modules = list(target_network.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:22]
            module3 = modules[22:]

            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 22: # googlenet caffe
            modules = list(target_network.children())
            module1 = modules[:22]
            module2 = modules[22:]

            model_1st = nn.Sequential(*[generator], *module1, Flatten())
            model_2nd = nn.Sequential(*module2)
        else:
            return None, None
    elif model_name == 'vgg19':
        if flat:
            layers = list(target_network.children())
            module1 = layers[:split_layer]
            module2 = layers[split_layer:]
            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*module2)
        else:
            if split_layer < 38:
                modules = list(target_network.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:split_layer]
                module2 = layers[split_layer:38]
                module3 = layers[38:]
                model_1st = nn.Sequential(*[generator], *module1)
                model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])
            else:
                modules = list(target_network.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:38]
                moduel2 = layers[38:split_layer]
                module3 = layers[split_layer:]
                model_1st = nn.Sequential(*[*[generator], *module1, Flatten(), *moduel2])
                model_2nd = nn.Sequential(*module3)
    elif model_name == 'alexnet':
        if split_layer == 6:
            modules = list(target_network.children())
            module1 = modules[0]
            module2 = [modules[1]]
            module_ = list(modules[2])
            module3 = module_[:5]
            module4 = module_[5:]

            model_1st = nn.Sequential(*[*[generator], *module1, Flatten(), *module2, *module3])
            model_2nd = nn.Sequential(*module4)
    elif model_name == 'shufflenetv2':
        if split_layer == 6:
            modules = list(target_network.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = modules[1:6]
            module2 = [sub_modules[0]]
            module3 = [sub_modules[1]]

            model_1st = nn.Sequential(*[*[generator], *module0, *module1, Avgpool2d_n(poolsize=7), Flatten(), *module2])
            model_2nd = nn.Sequential(*module3)
        elif split_layer == 1:
            modules = list(target_network.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = modules[2:6]
            module3 = sub_modules

            model_1st = nn.Sequential(*[*[generator], *module0, *module1])
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'mobilenet':
        if split_layer == 3:
            modules = list(target_network.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = [modules[3]]

            model_1st = nn.Sequential(*[*[generator], *module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(target_network.children())
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = [modules[2]]
            module3 = [modules[3]]
            model_1st = nn.Sequential(*[generator], *module0)
            model_2nd = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'wideresnet':
        if split_layer == 6:
            modules = list(target_network.children())
            module1 = modules[:2]
            module2 = modules[3:7]
            module3 = [modules[-1]]

            model_1st = nn.Sequential(*[*[generator], *module1, *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(target_network.children())
            module1 = modules[:2]

            module2 = modules[3:7]
            module3 = [modules[-1]]
            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    else:
        return None, None

    #summary(target_network, (3, 224, 224))
    #summary(model_1st, (3, 224, 224))

    return model_1st, model_2nd


def reconstruct_model(ori_model, model_name, mask, split_layer=43, rec_type='mask'):
    '''
    reconstruct filter model for uap generation
    Args:
        ori_model:
        model_name:
        mask:

    Returns:

    '''
    if rec_type == 'mask':
        if model_name == 'vgg19':
            if split_layer < 38:
                modules = list(ori_model.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:split_layer]
                module2 = layers[split_layer:38]
                module3 = layers[38:]
                mask = torch.reshape(mask, (64,112,112))

                # add mask
                model = nn.Sequential(*[*module1, Mask(mask), *module2, Flatten(), *module3])
                num_classes = 10
            else:
                modules = list(ori_model.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:38]
                moduel2 = layers[38:split_layer]
                module3 = layers[split_layer:]
                model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
                model_2nd = nn.Sequential(*module3)

                # add mask
                model = nn.Sequential(*[*module1, Flatten(), *moduel2, Mask(mask), *module3])
                num_classes = 10
        else:
            return None, 0
    elif rec_type == 'first':
        if model_name == 'vgg19':
            if split_layer < 38:
                return None, 0
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:38]
            moduel2 = layers[38:split_layer]
            module3 = layers[split_layer:]
            model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
            model = model_1st
            num_classes = 4096
        else:
            return None, 0

    return model, num_classes

def reconstruct_model_repair(ori_model, model_name, mask, split_layer=43):
    '''
    reconstruct filter model for uap generation
    Args:
        ori_model:
        model_name:
        mask:

    Returns:

    '''
    if model_name == 'vgg19':
        if split_layer < 38:
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:split_layer]
            module2 = layers[split_layer:38]
            module3 = layers[38:]
            mask = torch.reshape(mask, (64,112,112))

            # add mask
            model = nn.Sequential(*[*module1, Mask(mask), *module2, Flatten(), *module3])
            num_classes = 10
        else:
            modules = list(ori_model.children())
            layers = list(modules[0]) + [modules[1]] + list(modules[2])
            module1 = layers[:38]
            moduel2 = layers[38:split_layer]
            module3 = layers[split_layer:]
            model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
            model_2nd = nn.Sequential(*module3)

            # add mask
            model = nn.Sequential(*[*module1, Flatten(), *moduel2, Mask(mask), *module3])
            num_classes = 10
    else:
        return None, 0

    return model, num_classes


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1, p=2)[:,None,None]


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        x = F.relu(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Avgpool2d(nn.Module):
    def __init__(self):
        super(Avgpool2d, self).__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        return x


class Avgpool2d_n(nn.Module):
    def __init__(self, poolsize=2):
        super(Avgpool2d_n, self).__init__()
        self.poolsize = poolsize
    def forward(self, x):
        x = F.avg_pool2d(x, self.poolsize)
        return x


class MyAvgPool2D(nn.Module):
    '''
    tested model:
    - vgg19
    '''
    def __init__(self, output_size, replace=False):
        super(MyAvgPool2D, self).__init__()
        self.output_size = output_size
        self.kernel_size = 4
        self.stride = 4

    def forward(self, x):
        self.stride = x.size(2) // self.output_size[0]
        self.kernel_size = int(x.size(2) - (self.output_size[0] - 1) * self.stride)
        #print('[DEBUG] kernel_size {}'.format(self.kernel_size))
        #print('[DEBUG] stride {}'.format(self.stride))
        return self.get_pools(x)

    def get_pools(self, x):
        pooled = []
        for i in torch.arange(start=0, end=x.size(2), step=self.stride):
            for j in torch.arange(start=0, end=x.size(2), step=self.stride):
                #get a single pool
                fmap = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                pooled.append(torch.mean(fmap, dim=2))
        return torch.reshape(torch.stack(pooled, dim=2),
                             (pooled[0].size(0), pooled[0].size(1), self.output_size[0], self.output_size[1]))


class MyMaxPool2D(nn.Module):
    '''
    tested model:
    - vgg19
    '''
    def __init__(self, kernel_size, stride, replace=False):
        super(MyMaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = (7, 7)
        self.replace = replace

    def forward(self, x):
        self.output_size = (x.size(2) // self.stride, x.size(2) // self.stride)
        #print('[DEBUG] output_size {}'.format(self.output_size))
        if self.replace:
            return self.get_entropy_pools(self.get_pools(x))
        else:
            return self.get_pools(x)

    def get_pools(self, x):
        pooled = []
        for i in torch.arange(start=0, end=x.size(2), step=self.stride):
            for j in torch.arange(start=0, end=x.size(2), step=self.stride):
                #get a single pool
                fmap = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                pooled.append(torch.max(torch.max(fmap,dim=2).values,dim=2,keepdim=True).values)
        return torch.reshape(torch.stack(pooled, dim=2),
                             (pooled[0].size(0), pooled[0].size(1), self.output_size[0], self.output_size[1]))

    def get_entropy_pools(self, pooled):
        global_pool_entropy = F.softmax(pooled, dim=1) * F.log_softmax(pooled, dim=1)
        global_pool_entropy = -1.0 * global_pool_entropy.sum(dim=1)
        max_entropy = torch.unsqueeze(torch.max(
                                   torch.max(global_pool_entropy, dim=1).values, dim=1, keepdim=True).values, dim=2)
        #mask = (global_pool_entropy > max_entropy * 0.8)
        global_entropy_weight = (1 - torch.div(global_pool_entropy, max_entropy))
        weighted_pooled = pooled * torch.unsqueeze(global_entropy_weight, dim=1)
        #weighted_pooled = pooled * torch.unsqueeze(mask, dim=1)
        return weighted_pooled


class Mask(nn.Module):
    def __init__(self, mask):
        super(Mask, self).__init__()
        self.mask = mask.to(torch.float)
    def forward(self, x):
        x = x * self.mask
        return x


class EntropyWeight(nn.Module):
    def __init__(self):
        super(EntropyWeight, self).__init__()

    def forward(self, x):
        entropy = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        entropy = -1.0 * entropy.sum(dim=1)
        max_entropy = torch.max(entropy)
        adjust_weight = (1 - torch.div(entropy, max_entropy))
        x = x * torch.unsqueeze(adjust_weight, dim=1)
        return x


class ShannonEntropyBatch(nn.Module):
    def __init__(self, cuda=True):
        super(ShannonEntropyBatch, self).__init__()
        self.cuda = cuda

    # support per batch entropy calculation
    def forward(self, x):
        if x.dim() == 1:
            _, counts = torch.unique(x, return_counts=True)
            probabilities = counts / x.shape[-1]
            product = probabilities * torch.log2(probabilities)
        else:
            product = self.unique_rows(x)
        h = -torch.sum(product, dim=-1)
        return h

    def unique_rows(self, x):
        out = torch.zeros(x.shape)
        if self.cuda:
            out = out.cuda()
        for index, x_i in enumerate(x):
            _, counts = torch.unique(x_i, return_counts=True)
            probabilities = counts / x.shape[-1]
            product = probabilities * torch.log2(probabilities)
            out[index][:len(counts)] = product
        return out


class ShannonEntropyFlat(nn.Module):
    def __init__(self, cuda=True):
        super(ShannonEntropyFlat, self).__init__()
        self.cuda = cuda

    def forward(self, x):
        x = torch.flatten(x)
        _, counts = torch.unique(x, return_counts=True)
        probabilities = counts / len(x)
        product = probabilities * torch.log2(probabilities)
        h = -torch.sum(product, dim=-1)
        return h


class hloss(nn.Module):
    def __init__(self, cuda=True):
        super(hloss, self).__init__()
        self.cuda = cuda

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b