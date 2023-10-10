import torch
import time
import sys
from ipdb import set_trace

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger, l1_regularizer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    regu_losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=opt.is_amp):
        for i, (inputs, targets, clip_visfeas) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            if l1_regularizer is not None:
                reg_loss = l1_regularizer(model)
            else:
                reg_loss = torch.tensor(0)
            if not opt.ablation_removeOrig:
                inputs = inputs.cuda()
            targets = targets.cuda()
            if opt.is_w_knowledge:
                clip_visfeas = clip_visfeas.cuda()
            # if not opt.no_cuda:
            #     targets = targets.cuda(async=True)
            # inputs = Variable(inputs, volatile=True)
            # targets = Variable(targets, volatile=True)
            if opt.KnowAssistCLIPzs:
                outputs = model.module.knowAssist_CLIPzeroshot(inputs, clip_visfeas)
            else:
                if opt.fuse_mode=='no':
                    # from ipdb import set_trace;set_trace()
                    outputs = model(inputs, clip_visfeas)
                elif opt.fuse_mode=='cat':
                    outputs = model.module.cat_fuse_forward(inputs, clip_visfeas)

            loss = criterion(outputs, targets)
            if l1_regularizer is not None:
                loss += reg_loss
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data.item(), inputs.size(0))
            regu_losses.update(reg_loss.data.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            print_freq = len(data_loader)//10
            if i%(print_freq)==0:
                #set_trace()
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
        # from ipdb import set_trace;set_trace()
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

        return losses.avg
