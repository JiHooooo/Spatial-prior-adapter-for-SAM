import numpy as np
from utils import sod_metric 
import torch
import os
import shutil
import math
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import yaml

def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def calc_seg(y_pred, y_true):
    # y_pred: [b, n, h, w] batch_size, label_num, width, height
    # y_true: [b, n, h, w] batch_size, label_num, width, height
    # return s-measure, e-measure, mIoU, mDice
    batchsize = y_true.shape[0]
    metric_SM = sod_metric.Smeasure()
    metric_EM = sod_metric.Emeasure()
    metric_dice_iou = sod_metric.dice_iou_tool()
    with torch.no_grad():
        assert y_pred.shape == y_true.shape

        for i in range(batchsize):
            true_ori, pred_ori = \
                y_true[i, 0].cpu().data.numpy(), y_pred[i, 0].cpu().data.numpy()
            metric_dice_iou.step(pred=pred_ori, gt=true_ori)
            metric_SM.step(pred=pred_ori*255, gt=true_ori*255)
            metric_EM.step(pred=pred_ori*255, gt=true_ori*255)

        mDice, mIou = metric_dice_iou.get_results()
        sm = metric_SM.get_results()["sm"]
        em = metric_EM.get_results()["em"]["curve"].mean()
    

    return sm, em, mDice, mIou

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def set_log_path(path):
    global _log_path
    _log_path = path

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

def warmup_schedule(optimizer,
                        num_warmup_steps=1000,
                        last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            _lr = 1
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)