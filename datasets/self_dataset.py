import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2

import random
from .augmentation import get_augmentation, get_augmentation_gray

def xywh_to_xyxy(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0]+bbox[2]
    y2 = bbox[1]+bbox[3]
    return [x1,y1,x2,y2]

class PromiseDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, trainsize=1024, separate_info=None, label_num=1, train_flag = True, bbox_flag = True):
        self.trainsize = trainsize
        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.png')]
        # self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        # self.filter_files()
        self.label_num = label_num
        self.train_flag = train_flag
        self.trainsize = trainsize
        # 
        pd_info = pd.read_csv(separate_info)
        if train_flag:
            image_info = pd_info[pd_info['train']==1]
        else:
            image_info = pd_info[pd_info['train']==0]
        
        self.images = []
        self.gts = []
        for i in image_info.index:
            one_set_info = image_info.loc[i]
            self.images.append('%s/%s/%s/ori/%s'%(image_root, one_set_info['server_name'], 
                                              one_set_info['case_name'], one_set_info['image_name']))
            self.gts.append('%s/%s/%s/mask/%s'%(image_root, one_set_info['server_name'], 
                                              one_set_info['case_name'], one_set_info['mask_name']))
            
        print('number of dataset is %d'%(len(self.images)))
        self.size = len(self.images)
        
        self.transforms = get_augmentation_gray((trainsize, trainsize), train_flag=train_flag)
        self.bbox_flag = bbox_flag
    def __len__(self):
        return self.size
    
    def provide_bbox_prompt(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        find_bbox_flag = True
        while find_bbox_flag:
            transformed = self.transforms(image=image, mask=gt)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            #find the bbox
            # y_indices, x_indices = np.where(transformed_mask > 0)
            # x_min, x_max = np.min(x_indices), np.max(x_indices)
            # y_min, y_max = np.min(y_indices), np.max(y_indices)
            transformed_mask, bbox_xyxy_list = self.find_one_bbox(transformed_mask)
            if len(bbox_xyxy_list)>0:
                find_bbox_flag=False

        if not self.train_flag:
            bboxes = np.array(bbox_xyxy_list)
        else:
            bboxes_list = []
            # add perturbation to bounding box coordinates
            H, W = self.trainsize, self.trainsize
            for bbox_xyxy in bbox_xyxy_list:
                x_min = bbox_xyxy[0]
                x_max = bbox_xyxy[2]
                y_min = bbox_xyxy[1]
                y_max = bbox_xyxy[3]
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(W, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(H, y_max + np.random.randint(0, 20))
                bboxes_list.append([x_min, y_min, x_max, y_max])
            bboxes = np.array(bboxes_list)

        # convert img embedding, mask, bounding box to torch tensor
        # return self.image2tensor(transformed_image), self.mask2tensor_multi_bbox(transformed_mask, len(bbox_xyxy_list)), torch.tensor(bboxes).float()
        return {'image':self.image2tensor(transformed_image), 'mask':self.mask2tensor_multi_bbox(transformed_mask, len(bbox_xyxy_list)), 'box':torch.tensor(bboxes).float()}

    # each mask means one bbox
    def mask2tensor_multi_bbox(self, mask, mask_num):
        multi_mask = np.zeros((mask_num, mask.shape[0], mask.shape[1]), dtype=np.float32)
        for index_mask in range(mask_num):
            multi_mask[index_mask][mask == index_mask+1] = 1
        return (torch.from_numpy(multi_mask)).long()

    def provide_no_prompt(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        transformed = self.transforms(image=image, mask=gt)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_mask = cv2.resize(transformed_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        return self.image2tensor(transformed_image), self.mask2tensor(transformed_mask)

    def __getitem__(self, index):
        if self.bbox_flag:
            return self.provide_bbox_prompt(index)
        else:
            return self.provide_no_prompt(index)
    
    # def find_one_bbox(self,mask):
    #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     bbox_list = []
    #     self.contours_num = 0
    #     for contour in contours:
    #         # x y w h
    #         bbox_xywh = cv2.boundingRect(contour)
    #         if bbox_xywh[2]>1 and bbox_xywh[3]>1:
    #             bbox_xyxy = xywh_to_xyxy(bbox_xywh)
    #             bbox_list.append(bbox_xyxy)
    #             self.contours_num += 1
    #     if self.contours_num == 1:
    #         select_bbox = bbox_list[0]
    #         mask_new = mask
    #     else:    
    #         select_bbox = bbox_list[np.random.randint(len(bbox_list), size=1).item()]
    #         mask_new = np.zeros(mask.shape)
    #         mask_new[select_bbox[1]:select_bbox[3], select_bbox[0]:select_bbox[2]] = mask[select_bbox[1]:select_bbox[3], select_bbox[0]:select_bbox[2]]
    #     mask_new = cv2.resize(mask_new, (256, 256), interpolation=cv2.INTER_NEAREST)
    #     return mask_new, select_bbox
    def find_one_bbox(self, mask):

        self.contours_num = 0
        bbox_list = []

        area_num, area_image = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)
        for i in range(area_num-1):
            simple_mask = np.zeros(area_image.shape).astype(np.uint8)
            simple_mask[area_image==i+1] = 1
            contours,hierarchy = cv2.findContours(simple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_xywh = cv2.boundingRect(contours[0])
            if bbox_xywh[2] > 5 and bbox_xywh[3] > 5 and simple_mask.sum()>20:
                bbox_xyxy = xywh_to_xyxy(bbox_xywh)
                bbox_list.append(bbox_xyxy)
        mask_new = cv2.resize(area_image, (256, 256), interpolation=cv2.INTER_NEAREST)
        return mask_new, bbox_list

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts
        
    def rgb_loader(self, path):
        # with open(path, 'rb') as f:
        image = np.load(path)
        lower_bound, upper_bound = np.percentile(image, 0.5), np.percentile(image, 99.5)
        image_data_pre = np.clip(image, lower_bound, upper_bound).astype(np.float32)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))
        return np.concatenate([image_data_pre[:,:, np.newaxis], image_data_pre[:,:, np.newaxis], image_data_pre[:,:, np.newaxis]], axis=2)
    
    def binary_loader(self, path):
        mask_ori = np.load(path).astype(np.uint8)
        return mask_ori

    def image2tensor(self, image):
        # image = image / 255.0
        if len(image.shape) == 2:
            image = image[None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        else:
            raise ValueError('The size of input image should have two dimention or three dimention')
        return torch.from_numpy(image.astype(np.float32))

    def mask2tensor(self, mask):
        multi_mask = np.zeros((self.label_num, mask.shape[0], mask.shape[1]), dtype=np.float32)
        for index_mask in range(self.label_num):
            multi_mask[index_mask][mask == index_mask+1] = 1
        return (torch.from_numpy(multi_mask)).long()


class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_root, gt_root, trainsize=1024, separate_info=None, label_num=1, train_flag = True, thresh=127, bbox_flag=True):
        self.trainsize = trainsize
        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.png')]
        # self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        
        self.thresh = thresh
        self.label_num = label_num
        self.train_flag = train_flag
        self.trainsize = trainsize
        # 
        pd_info = pd.read_csv(separate_info)
        if train_flag:
            image_info = pd_info[pd_info['train']==1]
        else:
            image_info = pd_info[pd_info['train']==0]
        self.images = ['%s/%s'%(image_root, i) for i in image_info['image']]
        image_basenames = [os.path.splitext(i)[0] for i in image_info['image']]
        self.gts = ['%s/%s.png'%(gt_root, i) for i in image_basenames]
        # self.filter_files()
        print('number of dataset is %d'%(len(self.images)))
        self.size = len(self.images)
        
        self.transforms = get_augmentation((trainsize, trainsize), train_flag=train_flag)

        self.bbox_flag = bbox_flag
    
    def provide_bbox_prompt(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        find_bbox_flag = True
        while find_bbox_flag:
            transformed = self.transforms(image=image, mask=gt)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            #find the bbox
            # y_indices, x_indices = np.where(transformed_mask > 0)
            # x_min, x_max = np.min(x_indices), np.max(x_indices)
            # y_min, y_max = np.min(y_indices), np.max(y_indices)
            transformed_mask, bbox_xyxy_list = self.find_one_bbox(transformed_mask)
            if len(bbox_xyxy_list) > 0:
                find_bbox_flag=False
                
        if not self.train_flag:
            bboxes = np.array(bbox_xyxy_list)
        else:
            bboxes_list = []
            # add perturbation to bounding box coordinates
            H, W = self.trainsize, self.trainsize
            for bbox_xyxy in bbox_xyxy_list:
                x_min = bbox_xyxy[0]
                x_max = bbox_xyxy[2]
                y_min = bbox_xyxy[1]
                y_max = bbox_xyxy[3]
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(W, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(H, y_max + np.random.randint(0, 20))
                bboxes_list.append([x_min, y_min, x_max, y_max])
            bboxes = np.array(bboxes_list)

        # convert img embedding, mask, bounding box to torch tensor
        # image: [C, W, h], mask: [N, W, H], bbox: [N, 4]
        return {'image':self.image2tensor(transformed_image), 'mask':self.mask2tensor_multi_bbox(transformed_mask, len(bbox_xyxy_list)), 'box':torch.tensor(bboxes).float()}

    def provide_no_prompt(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        transformed = self.transforms(image=image, mask=gt)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_mask = cv2.resize(transformed_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        return self.image2tensor(transformed_image), self.mask2tensor(transformed_mask)

    def __getitem__(self, index):
        if self.bbox_flag:
            return self.provide_bbox_prompt(index)
        else:
            return self.provide_no_prompt(index)
    
    def find_one_bbox(self,mask):

        self.contours_num = 0
        bbox_list = []

        area_num, area_image = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)
        for i in range(area_num-1):
            simple_mask = np.zeros(area_image.shape).astype(np.uint8)
            simple_mask[area_image==i+1] = 1
            contours,hierarchy = cv2.findContours(simple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_xywh = cv2.boundingRect(contours[0])
            if bbox_xywh[2] > 5 and bbox_xywh[3] > 5 and simple_mask.sum()>20:
                bbox_xyxy = xywh_to_xyxy(bbox_xywh)
                bbox_list.append(bbox_xyxy)
        mask_new = cv2.resize(area_image, (256, 256), interpolation=cv2.INTER_NEAREST)
        return mask_new, bbox_list


    # def filter_files(self):
    #     assert len(self.images) == len(self.gts)
    #     images = []
    #     gts = []
    #     for img_path, gt_path in zip(self.images, self.gts):
    #         img = Image.open(img_path)
    #         gt = Image.open(gt_path)
    #         if img.size == gt.size:
    #             images.append(img_path)
    #             gts.append(gt_path)
    #     self.images = images
    #     self.gts = gts
        
    def rgb_loader(self, path):
        # with open(path, 'rb') as f:
        img = Image.open(path)
        return np.array(img).astype(np.uint8)

    def binary_loader(self, path):
        mask_ori = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask_new = np.zeros(mask_ori.shape).astype(np.uint8)
        mask_new[mask_ori>self.thresh]=1
            # return img.convert('1')
        return mask_new

    def image2tensor(self, image):
        # image = image / 255.0
        if len(image.shape) == 2:
            image = image[None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        else:
            raise ValueError('The size of input image should have two dimention or three dimention')
        return torch.from_numpy(image.astype(np.float32))

    # each mask means one bbox
    def mask2tensor_multi_bbox(self, mask, mask_num):
        multi_mask = np.zeros((mask_num, mask.shape[0], mask.shape[1]), dtype=np.float32)
        for index_mask in range(mask_num):
            multi_mask[index_mask][mask == index_mask+1] = 1
        return (torch.from_numpy(multi_mask)).long()

    # each mask means one label
    def mask2tensor(self, mask):
        multi_mask = np.zeros((self.label_num, mask.shape[0], mask.shape[1]), dtype=np.float32)
        for index_mask in range(self.label_num):
            multi_mask[index_mask][mask == index_mask+1] = 1
        return (torch.from_numpy(multi_mask)).long()
    # def convert2polar(self, img, gt):
    
    #     center = polar_transformations.centroid(gt)
    #     img = polar_transformations.to_polar(img, center)
    #     gt = polar_transformations.to_polar(gt, center)
    	
    #     return img, gt
    
    # def resize(self, img, gt):
    #     assert img.size == gt.size
    #     w, h = img.size
    #     if h < self.trainsize or w < self.trainsize:
    #         h = max(h, self.trainsize)
    #         w = max(w, self.trainsize)
    #         return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
    #     else:
    #         return img, gt

    def __len__(self):
        return self.size
    


class FubaoDataset(PolypDataset):
    """
    dataloader for Fubao ultrasound tumor segmentation
    """
    def __init__(self, image_root, trainsize=1024, separate_info=None, label_num=1, train_flag=True, thresh=127, bbox_flag=True):
       
        self.trainsize = trainsize
        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.png')]
        # self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        # self.filter_files()
        self.thresh = thresh
        self.label_num = label_num
        self.train_flag = train_flag
        self.trainsize = trainsize
        # 
        if isinstance(separate_info, str):

            pd_info = pd.read_csv(separate_info)
            if train_flag:
                image_info = pd_info[pd_info['train']==1]
            else:
                image_info = pd_info[pd_info['train']==0]
            self.images = ['%s%s'%(image_root, i) for i in image_info['image']]
            self.gts =  ['%s%s'%(image_root, i) for i in image_info['mask']]
            print('number of dataset is %d'%(len(self.images)))
        elif isinstance(separate_info, list):
            self.images = []
            self.gts = []
            for i in range(len(separate_info)):
                pd_info = pd.read_csv(separate_info[i])
                if train_flag:
                    image_info = pd_info[pd_info['train']==1]
                else:
                    image_info = pd_info[pd_info['train']==0]
                images_one = ['%s%s'%(image_root, i) for i in image_info['image']]
                gts_one =  ['%s%s'%(image_root, i) for i in image_info['mask']]
                self.images.extend(images_one)
                self.gts.extend(gts_one)
                print('number of dataset(%s) is %d'%(separate_info[i], len(images_one)))
        ####
        print('total number of dataset() is %d'%(len(self.images)))
        self.size = len(self.images)
        
        self.transforms = get_augmentation((trainsize, trainsize), train_flag=train_flag)

        self.bbox_flag = bbox_flag

    def binary_loader(self, path):
        mask_ori = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask_new = np.zeros(mask_ori.shape).astype(np.uint8)
        mask_new[mask_ori>self.thresh]=1
            # return img.convert('1')
        return mask_new