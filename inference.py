import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from local_segment_anything import SamPredictor, sam_model_registry
from local_segment_anything.utils.transforms import ResizeLongestSide
import argparse
#self dataset
from datasets.self_dataset import PolypDataset, PromiseDataset
from utils.tools import compute_num_params
from utils.tools import set_save_path
from utils.tools import calc_seg
import tqdm as tq
import yaml
from PIL import Image
import cv2


############### 参数设定  ###################
config_file = "/media/gallade/RAID/hu/code_lib/sam_spa/work_dir_poly/poly_test/setting.yaml"
pretrained_model = "/media/gallade/RAID/hu/code_lib/sam_spa/work_dir_poly/poly_test/sam_model_best.pth"
save_folder = "evaluation_results"

###########################################

def val_show(sam_model, dataset_val, save_folder):
        
    sam_model.eval()
    
    for step, one_loader in enumerate(tq.tqdm(dataset_val, desc='valid')):
        with torch.no_grad():
            # get image feature

            image = one_loader['image']
            gt2D = one_loader['mask']
            boxes = one_loader['box']
            image_embedding = sam_model.image_encoder(image[None,:].to(device))
            num_bbox = boxes.shape[0]
            image_embedding = image_embedding.repeat(num_bbox,1,1,1)
            
            box_torch = boxes[None, :].permute(1,0,2).to(device)
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            
            # get mask result
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            
            pred_mask_all = np.zeros((256,256))
            pred_result = torch.sigmoid(low_res_masks).cpu().numpy()
            for box_index in range(num_bbox):
                gt2D_one = gt2D[box_index][None,:,:].numpy()
                
                pred_binary = np.zeros(pred_result[box_index].shape).astype(np.uint8)
                pred_binary[pred_result[box_index]>0.5] = 1

                image_path = dataset_val.images[step]
                image_name = os.path.split(image_path)[1]
                base_name = os.path.splitext(image_name)[0]

                pred_mask_all[pred_binary[0]==1]=1
            #save_image
            pred_mask = (pred_mask_all*255).astype(np.uint8)
            
            Image.fromarray(pred_mask).save('%s/%s.png'%(save_folder, base_name))
        

def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

class args_empty(object):
    def _init_(self):
        pass

# load parameter from config file
args = args_empty()
over_write_args_from_file(args, config_file)
args.checkpoint = None
device = torch.device('cuda:0')

# build model
if args.adapter and args.adapter_type == 'spa':
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, adapter_flag=args.adapter, \
                                                adapter_type=args.adapter_type, prompt_flag=args.prompt_flag, \
                                                deform_num_heads=args.deform_num_heads, attn_type=args.attn_type, \
                                                interaction_indexes=args.interaction_indexes, n_points=args.n_points, \
                                                init_values=args.init_values, drop_path=args.drop_path, \
                                                with_cffn=args.with_cffn, cffn_ratio=args.cffn_ratio, \
                                                deform_ratio=args.deform_ratio, with_cp=args.with_cp, \
                                                n_level_list=args.n_level_list).to(device)
else:
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, adapter_flag=args.adapter, \
                                                adapter_type=args.adapter_type, \
                                            prompt_flag=args.prompt_flag).to(device)

# load pretrainede model parameter 
with open(pretrained_model, 'rb') as f:
    state_dict = torch.load(f)
sam_model.load_state_dict(state_dict, strict=False)
sam_model.eval()

# build folder to save inference results
if os.path.exists(save_folder):
    pass
else:
    os.makedirs(save_folder)

# load dataloader
valid_dataset = PolypDataset(image_root=args.image_root, gt_root=args.gt_root, separate_info=args.separate_info, train_flag=False, bbox_flag=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# inference processs
val_show(sam_model, valid_dataset, save_folder)