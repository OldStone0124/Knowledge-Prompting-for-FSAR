import torch
import torch.nn as nn
import time
import os
import sys
from ipdb import set_trace

from utils import AverageMeter, calculate_accuracy, check_which_nan, check_tensor_nan
l1_crit = nn.L1Loss(size_average=False)

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, yfcc_loader=None, l1_regularizer=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    regu_losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    if yfcc_loader is not None:
        yfcc_loader_iter = iter(yfcc_loader)

    for i, (inputs, targets, clip_visfeas) in enumerate(data_loader):
        # https://discuss.pytorch.org/t/simple-l2-regularization/139/2?u=zhi_li
        if l1_regularizer is not None:
            reg_loss = l1_regularizer(model)
            # from ipdb import set_trace;set_trace()
        else:
            reg_loss = torch.tensor(0)

        data_time.update(time.time() - end_time)

        # if not opt.no_cuda:
        #     targets = targets.cuda(async=True)
        # inputs = Variable(inputs)
        # targets = Variable(targets)
        with torch.cuda.amp.autocast(enabled=opt.is_amp):
            if yfcc_loader is not None:
                try:
                    yfcc_inputs, yfcc_targets = next(yfcc_loader_iter)
                except StopIteration:
                    yfcc_loader_iter = iter(yfcc_loader)
                    yfcc_inputs, yfcc_targets = next(yfcc_loader_iter)

                inputs = torch.cat((inputs, yfcc_inputs), 0)
                targets = torch.cat((targets, yfcc_targets), 0)
            #inputs =>  torch.Size([32, 3, 16, 112, 112])
            # outputs => torch.Size([32, 64])
            # import ipdb;ipdb.set_trace()
            if not opt.ablation_removeOrig:
                inputs = inputs.cuda()
            targets = targets.cuda()
            #set_trace()
            if opt.is_w_knowledge:
                clip_visfeas = clip_visfeas.cuda()
            if opt.KnowAssistCLIPzs:
                outputs = model.module.knowAssist_CLIPzeroshot(inputs, clip_visfeas)
            else:
                if opt.fuse_mode=='no':
                    outputs = model(inputs, clip_visfeas)
                elif opt.fuse_mode=='cat':
                    outputs = model.module.cat_fuse_forward(inputs, clip_visfeas)
            # if 'FLOPs' in opt.result_path:
            #     # from lin_utils import flops
            #     # set_trace()
            #     # flops(model, (inputs, clip_visfeas))
            #     from thop import profile
            #     set_trace()
            #     flops, params = profile(model, (inputs, clip_visfeas))

            #print(outputs.size())
            #if opt.model == 'avts':

            loss = criterion(outputs, targets)
            if l1_regularizer is not None:
                loss += reg_loss
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data.item(), inputs.size(0))
            # from ipdb import set_trace;set_trace()
            regu_losses.update(reg_loss.data.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

        if opt.check_nan and check_which_nan(model, is_break=True):
            set_trace()
            check_which_nan(model)
        optimizer.zero_grad()
        opt.scaler.scale(loss).backward()
        # optimizer.step()
        opt.scaler.step(optimizer)
        opt.scaler.update()
        if opt.check_nan and check_which_nan(model, is_break=True):
            set_trace()
            check_which_nan(model)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # from ipdb import set_trace;set_trace()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        if i%(opt.print_freq)==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'l1reguLoss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    reg_loss=regu_losses,
                    acc=accuracies))
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
