# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from local_segment_anything import SamPredictor, sam_model_registry
from local_segment_anything.utils.transforms import ResizeLongestSide
import argparse
#self dataset
from datasets.self_dataset import PolypDataset, PromiseDataset, FubaoDataset
from utils.tools import compute_num_params, warmup_schedule, get_cosine_schedule_with_warmup
from utils.tools import set_save_path, over_write_args_from_file
from utils.tools import calc_seg
from utils.engine import train_engine_with_bbox_prompt, val_engine_with_bbox_prompt, \
                        train_engine_no_prompt, val_engine_no_prompt
from utils.loss import FocalLoss


# %% set up parser
parser = argparse.ArgumentParser()
# config file
parser.add_argument('--c', type=str, default='')
parser.add_argument('--data_type', type=str, default='polyp')
parser.add_argument('--task_name', type=str, default='SAM-ViT-B')
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--checkpoint', type=str, default='/home/hjh/code_lib/SAM/sam_vit_b_01ec64.pth')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--work_dir', type=str, default='./work_dir')
# train
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--eval_epoch', type=int, default=20)
parser.add_argument('--warm_iter', type=int, default=0)
# batch_size must be 1
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--seed', type=int, default=0)

#adapter
parser.add_argument('--image_root', type=str, default='/home/hjh/data_disk/hjh/dataset/Kvasir-SEG/images')
parser.add_argument('--gt_root', type=str, default='/home/hjh/data_disk/hjh/dataset/Kvasir-SEG/masks_new')
parser.add_argument('--separate_info', type=str, default='/home/hjh/data_disk/hjh/dataset/Kvasir-SEG/fold_info/kvasir_fold_0.csv')
parser.add_argument('--adapter', type=bool, default=False)
parser.add_argument('--finetune_all', type=bool, default=False)
args = parser.parse_args()
# load args from config file
over_write_args_from_file(args, args.c)
#
# set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
import random
random.seed(args.seed)
# batch_size = 1
batch_size = args.batch_size
batch_size_val = 1
# %% set up model for fine-tuning 
device = args.device
model_save_path = join(args.work_dir, args.task_name)
os.makedirs(model_save_path, exist_ok=True)

# save setting config file
save_setting_file_path = '%s/setting.yaml'%(model_save_path)
shutil.copy(args.c, save_setting_file_path)

# load log tool and tensorboard writer tool
log, writer = set_save_path(model_save_path, remove=False)

# load model
if args.adapter and args.adapter_type == 'spa':
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, adapter_flag=args.adapter, 
                                                    adapter_type=args.adapter_type, prompt_flag=args.prompt_flag,
                                                    deform_num_heads=args.deform_num_heads, attn_type=args.attn_type,
                                                    interaction_indexes=args.interaction_indexes, n_points=args.n_points,
                                                    init_values=args.init_values, drop_path=args.drop_path,
                                                    with_cffn=args.with_cffn, cffn_ratio=args.cffn_ratio,
                                                    deform_ratio=args.deform_ratio, with_cp=args.with_cp,
                                                    n_level_list=args.n_level_list).to(device)
else:
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, adapter_flag=args.adapter, adapter_type=args.adapter_type,
                                                prompt_flag=args.prompt_flag).to(device)
sam_model.train()

# Set up the optimizer, hyperparameter tuning will improve performance here
for name, para in sam_model.named_parameters():
    if 'mask_decoder' in name:
        para.requires_grad_(True)
    elif 'prompt_encoder' in name:
        para.requires_grad_(False)
    elif "image_encoder" in name and ("interaction" not in name and 'SPM' not in name and 'segmentic_token' not in name and 'prompt_generator' not in name and 'Adapter' not in name):
        if args.finetune_all:
            para.requires_grad_(True)
        else:
            para.requires_grad_(False)
    else:
        para.requires_grad_(True)

# input_test = torch.randn(1, 3, 1024, 1024).to(device)
# flops, params = profile(sam_model.image_encoder, inputs=(input_test,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

if args.adapter_type == 'spa' and args.adapter:
    for name, para in sam_model.named_parameters():
        if 'neck' in name and 'image_encoder' in name:
            para.requires_grad_(True)

log('model: #params={}'.format(compute_num_params(sam_model, text=True)))
model_total_params = sum(p.numel() for p in sam_model.parameters())
model_grad_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
log('model_grad_params: %.2f M'%(model_grad_params/1e6))
log('model_total_params: %.2f M'%(model_total_params/1e6))


seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# regress loss for IoU/DSC prediction; (ignored for simplicity but will definitely included in the near future)
# regress_loss = torch.nn.MSELoss(reduction='mean')
# ce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
ce_loss = FocalLoss()

# check whether to output bbox prompt
if args.prompt_flag == 'bbox':
    bbox_flag = True
else:
    bbox_flag = False
# load dataset
if args.data_type == 'polyp':
    train_dataset = PolypDataset(image_root=args.image_root, gt_root=args.gt_root, separate_info=args.separate_info, train_flag=True, bbox_flag=bbox_flag)
    valid_dataset = PolypDataset(image_root=args.image_root, gt_root=args.gt_root, separate_info=args.separate_info, train_flag=False, bbox_flag=bbox_flag)
elif args.data_type == 'prom':
    train_dataset = PromiseDataset(image_root=args.image_root, separate_info=args.separate_info, train_flag=True, bbox_flag=bbox_flag)
    valid_dataset = PromiseDataset(image_root=args.image_root, separate_info=args.separate_info, train_flag=False, bbox_flag=bbox_flag)
elif args.data_type == 'fubao':
    train_dataset = FubaoDataset(image_root=args.image_root, separate_info=args.separate_info, train_flag=True, bbox_flag=bbox_flag)
    valid_dataset = FubaoDataset(image_root=args.image_root, separate_info=args.separate_info, train_flag=False, bbox_flag=bbox_flag)

def collate_fn_multi_batch(data_batch):
    # solve the problem of different bbox
    batch_image = torch.stack([item['image'] for item in data_batch], dim=0)
    batch_mask = [item['mask'] for item in data_batch]
    batch_bbox_list = [item['box'] for item in data_batch]
    result = dict()
    result['image'] = batch_image
    result['mask'] = batch_mask
    result['box'] = batch_bbox_list
    return result
if bbox_flag:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn_multi_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size_val, shuffle=False, num_workers=8, collate_fn=collate_fn_multi_batch)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size_val, shuffle=False, num_workers=4)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, sam_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
# lr_schedule = warmup_schedule(optimizer, num_warmup_steps=args.warm_iter)

num_training_step = len(train_dataloader) * args.num_epochs
lr_schedule = get_cosine_schedule_with_warmup(optimizer,num_training_steps= num_training_step, 
                                              num_warmup_steps=args.warm_iter, )

#%% train
num_epochs = args.num_epochs
losses = []
best_dice = 0
for epoch in range(num_epochs):
    # train one epoch
    if args.prompt_flag == 'bbox':
        epoch_loss = train_engine_with_bbox_prompt(train_dataloader, sam_model, optimizer, lr_schedule, seg_loss, ce_loss, device)
    else:
        epoch_loss = train_engine_no_prompt(train_dataloader, sam_model, optimizer, lr_schedule, seg_loss, ce_loss, device, prompt_flag=args.prompt_flag)
    losses.append(epoch_loss)
    log(f'EPOCH: {epoch}, Loss: {epoch_loss}')

    # draw picture
    # %% plot loss
    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show() # comment this line if you are running on a server
    plt.savefig(join(model_save_path, 'train_loss.png'))
    plt.close()
    # save the model checkpoint
    
    # model evaluation    
    if epoch % args.eval_epoch == 0:

        sam_model.eval()
        
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))

        if args.prompt_flag == 'bbox':
            sm, em, mDice, mIoU = val_engine_with_bbox_prompt(valid_dataloader, sam_model, calc_seg, device)
        else:
            sm, em, mDice, mIoU = val_engine_no_prompt(valid_dataloader, sam_model, calc_seg, device, prompt_flag=args.prompt_flag)
        log('epoch %d validation results: SM(%.4f) EM(%.4f) mDice(%.2f) mIoU(%.2f)'%(epoch, sm, em, mDice*100, mIoU*100))
        
        # save best model
        if mDice > best_dice:
            best_dice = mDice
            torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))
            log('find a best model under epoch %d with dice %.2f '%(epoch, mDice*100))

        sam_model.train() 
print('results folder : %s'%(model_save_path))