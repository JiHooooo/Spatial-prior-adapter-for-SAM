import torch
from tqdm import tqdm

def custom_repeat(tensor, repeat_counts):
    assert len(tensor) == len(repeat_counts)
    repeated_tensors = [tensor[i].repeat(repeat_count, 1, 1, 1) for i, repeat_count in enumerate(repeat_counts)]
    return torch.concat(repeated_tensors, dim=0)

def train_engine_no_prompt(dataloader, model, optimizer, lr_schedule, dice_loss_func, ce_loss_func, device, desc_tqdm='train', prompt_flag='simple'):
    epoch_loss = 0
    for step, (image, gt2D) in enumerate(tqdm(dataloader, desc=desc_tqdm)):
        # image [1, 3, 1024, 1024]
        # gt2d [1, N, 256, 256] N means the number of bbox
        # boxes [1, N, 4]
        bs = image.shape[0]
        image_embedding = model.image_encoder(image.to(device))
        
        sparse_embeddings = torch.empty((bs, 0, model.prompt_encoder.embed_dim), device=model.prompt_encoder._get_device())
        dense_embeddings = model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, model.prompt_encoder.image_embedding_size[0], model.prompt_encoder.image_embedding_size[1]
        )
        if prompt_flag == 'simple':
            segmentic_token = model.image_encoder.segmentic_token.weight.unsqueeze(0).expand(bs, -1, -1)
            sparse_embeddings = torch.cat((sparse_embeddings, segmentic_token), dim=1)

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )
        
        dice_loss = dice_loss_func(low_res_masks, gt2D.permute(1,0,2,3).to(device)) 
        ce_loss = ce_loss_func(low_res_masks, gt2D.permute(1,0,2,3).float().to(device))
        loss = dice_loss + ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        epoch_loss += loss.item()
    epoch_loss /= (step+1)
    return epoch_loss

@torch.no_grad()
def val_engine_no_prompt(dataloader, model, eval_func, device, desc_tqdm='valid', prompt_flag='simple'):
    pred_list = []
    gt_list = []
    for step, (image, gt2D) in enumerate(tqdm(dataloader, desc=desc_tqdm)):
        # image [1, 3, 1024, 1024]
        # gt2d [1, N, 256, 256] N means the number of bbox
        bs = image.shape[0]
        image_embedding = model.image_encoder(image.to(device))
        
        sparse_embeddings = torch.empty((bs, 0, model.prompt_encoder.embed_dim), device=model.prompt_encoder._get_device())
        dense_embeddings = model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, model.prompt_encoder.image_embedding_size[0], model.prompt_encoder.image_embedding_size[1]
        )
        if prompt_flag == 'simple':
            segmentic_token = model.image_encoder.segmentic_token.weight.unsqueeze(0).expand(bs, -1, -1)
            sparse_embeddings = torch.cat((sparse_embeddings, segmentic_token), dim=1)
        
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )
        
        pred_list.append(torch.sigmoid(low_res_masks))
        gt_list.append(gt2D.permute(1,0,2,3))
    
    pred_list = torch.cat(pred_list, 0)
    gt_list = torch.cat(gt_list, 0)

    sm, em, mDice, mIoU = eval_func(pred_list, gt_list)
    return sm, em, mDice, mIoU

def train_engine_with_bbox_prompt(dataloader, model, optimizer, lr_schedule, dice_loss_func, ce_loss_func, device, desc_tqdm='train'):
    epoch_loss = 0
    for step, one_loader in enumerate(tqdm(dataloader, desc=desc_tqdm)):
        # image [B, 3, 1024, 1024]
        # gt2d list of [N, 256, 256] N means the number of bbox
        # boxes list of [N, 4]
        image = one_loader['image']
        gt2D = one_loader['mask']
        boxes = one_loader['box']
        num_bbox_list = [i.shape[0] for i in boxes]
        image_embedding = model.image_encoder(image.to(device))
        image_embedding = custom_repeat(image_embedding, num_bbox_list)
        gt2D = torch.concat(gt2D, dim=0)[:, None, :].to(device)
        with torch.no_grad():
            # convert box to 1024x1024 grid
            # box_np = boxes.numpy()
            # sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            # box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            # box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            # box_torch = boxes.to(device)
            # if len(box_torch.shape) == 2:
            #     box_torch = box_torch[:, None, :] # (B, 1, 4)
            # else:
            #     box_torch = box_torch.permute(1,0,2)
            box_torch = torch.concat(boxes, dim=0)[:, None, :].to(device)
            # dense embedding: [bs, 256, 64, 64]
            # sparse embedding: bbox [bs, 2, 256]

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None, #
                boxes=box_torch, # [N, 1, 4]
                masks=None, # [1, ]
            )
        
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )
        
        dice_loss = dice_loss_func(low_res_masks, gt2D) 
        # ce_loss = ce_loss_func(low_res_masks, gt2D.permute(1,0,2,3).float().to(device))
        # loss = dice_loss + ce_loss *20.0
        loss = dice_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        epoch_loss += loss.item()
    epoch_loss /= (step+1)
    return epoch_loss

@torch.no_grad()
def val_engine_with_bbox_prompt(dataloader, model, eval_func, device, desc_tqdm='valid'):
    pred_list = []
    gt_list = []
    for step, one_loader in enumerate(tqdm(dataloader, desc=desc_tqdm)):
        # image [B, 3, 1024, 1024]
        # gt2d [B, N, 256, 256] N means the number of bbox
        # boxes list of [N, 4]
        image = one_loader['image']
        gt2D = one_loader['mask']
        boxes = one_loader['box']
        num_bbox_list = [i.shape[0] for i in boxes]
        image_embedding = model.image_encoder(image.to(device))
        image_embedding = custom_repeat(image_embedding, num_bbox_list)
        gt2D = torch.concat(gt2D, dim=0)[:, None, :].to(device)
        # image [1, 3, 1024, 1024]
        # gt2d [1, N, 256, 256] N means the number of bbox
        # boxes [1, N, 4]
        # num_bbox = boxes.shape[1]
        # image_embedding = model.image_encoder(image.to(device))
        # image_embedding = image_embedding.repeat(num_bbox,1,1,1)
            # convert box to 1024x1024 grid
            # box_np = boxes.numpy()
            # sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            # box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            # box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        # box_torch = boxes.to(device)
        # if len(box_torch.shape) == 2:
        #     box_torch = box_torch[:, None, :] # (B, 1, 4)
        # else:
        #     box_torch = box_torch.permute(1,0,2)
        box_torch = torch.concat(boxes, dim=0)[:, None, :].to(device)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )
        
        pred_list.append(torch.sigmoid(low_res_masks))
        gt_list.append(gt2D)
    
    pred_list = torch.cat(pred_list, 0)
    gt_list = torch.cat(gt_list, 0)

    sm, em, mDice, mIoU = eval_func(pred_list, gt_list)
    return sm, em, mDice, mIoU