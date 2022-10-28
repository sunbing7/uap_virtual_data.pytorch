from __future__ import division
import numpy as np
import os, shutil, time
import itertools
import torch
import torch.nn as nn

from utils.utils import time_string, print_log


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
    while (iteration<num_iterations):
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
        if clean_acc.avg !=0:
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

def solve_causal(data_loader, filter_model, uap, filter_arch, target_class, num_sample, log=None, use_cuda=True):
    '''
    perform causality analysis on the dense layer before logit layer
    Args:
        data_loader: loader that loads original images with uap
        filter_model:
        uap:
        filter_arch:
        target_class:
        num_sample: number of samples to use for causality analysis
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
    out = []
    do_predict_avg = []
    for input, gt in data_loader:
        if total_num_samples >= num_sample:
            break
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()
            uap = uap.cuda()

        # compute output
        with torch.no_grad():
            dense_output = model1(input + uap)
            ori_output = model2(dense_output)
            #ori_output = filter_model(input)
            do_predict_neu = []
            do_predict = []
            #do convention for each neuron
            for i in range(0, len(dense_output[0])):
                hidden_do = np.zeros(shape=dense_output[:, i].shape)
                dense_output_ = torch.clone(dense_output)
                dense_output_[:, i] = torch.from_numpy(hidden_do)
                output_do = model2(dense_output_).cpu().detach().numpy()
                do_predict_neu.append(output_do) # 4096x32x10
            do_predict_neu = np.array(do_predict_neu)
            do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
            do_predict = np.mean(np.array(do_predict_neu), axis=1)  #4096x10
            #do_predict = do_predict_neu[:,target_class]

        do_predict_avg.append(do_predict) #batchx4096x11
        total_num_samples += len(gt)
    # average of all baches
    do_predict_avg = np.mean(np.array(do_predict_avg), axis=0) #4096x10
    # insert neuron index
    idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
    do_predict_avg = np.c_[idx, do_predict_avg]

    return do_predict_avg


def eval_uap(test_data_loader, target_model, uap, target_class, log=None, use_cuda=True):


    # switch to evaluate mode
    target_model.eval()

    total_num_samples = 0
    num_attack_success = 0
    '''
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
    clean_correctly_classified = 0
    for input, gt in test_data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()
            uap = uap.cuda()

        # compute output
        with torch.no_grad():
            attack_output = target_model(input + uap)
            ori_output = target_model(input)

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(ori_output, dim=-1).cpu().numpy()
        pert_out_class = torch.argmax(attack_output, dim=-1).cpu().numpy()

        clean_correctly_classified += np.sum(clean_out_class == gt.cpu().numpy())
        num_attack_success += np.sum(pert_out_class == target_class)

        total_num_samples += len(gt)
    test_sr = num_attack_success / total_num_samples * 100
    clean_test_acc = clean_correctly_classified / total_num_samples * 100
    return test_sr, clean_test_acc


def split_model(ori_model, model_name):
    '''
    split given model from the dense layer before logits
    Args:
        ori_model:
        model_name: model name
    Returns:
        splitted models
    '''
    if model_name == 'vgg19':
        modules = list(ori_model.children())
        layers = list(modules[0]) + [modules[1]] + list(modules[2])
        module1 = layers[:38]
        moduel2 = layers[38:43]
        module3 = layers[43:]
        model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
        model_2nd = nn.Sequential(*module3)
    else:
        return None, None

    return model_1st, model_2nd


def reconstruct_model(ori_model, model_name, mask):
    '''
    reconstruct filter model for uap generation
    Args:
        ori_model:
        model_name:
        mask:

    Returns:

    '''
    modules = list(ori_model.children())
    layers = list(modules[0]) + [modules[1]] + list(modules[2])
    module1 = layers[:38]
    moduel2 = layers[38:43]
    module3 = layers[43:]
    model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
    model_2nd = nn.Sequential(*module3)

    if model_name == 'vgg19':
        modules = list(ori_model.children())
        layers = list(modules[0]) + [modules[1]] + list(modules[2])
        module1 = layers[:38]
        moduel2 = layers[38:43]
        module3 = layers[43:]
        model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
        model_2nd = nn.Sequential(*module3)

        # add mask
        model = nn.Sequential(*[*module1, Flatten(), *moduel2, Mask(mask), *module3])
    else:
        return None

    return model


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


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Mask(nn.Module):
    def __init__(self, mask):
        super(Mask, self).__init__()
        self.mask = mask.to(torch.float)
    def forward(self, x):
        x = x * self.mask
        return x