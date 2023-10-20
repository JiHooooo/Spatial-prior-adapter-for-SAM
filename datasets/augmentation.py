import albumentations as A
#Convert image to torch.Tensor and divide by 255 if image or mask are uint8 type.
# from albumentations.pytorch import ToTensor
import cv2
import numpy as np

 # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
def get_augmentation_gray(image_size, train_flag=True):
    #image_size tuple or list of [height, width]
    small_len = min(image_size)
    argument_list = []
    if train_flag:
        argument_list.extend([
            A.Resize(height=image_size[0], width=image_size[1], p=1.0),
            # A.RandomResizedCrop(height=image_size[0], width=image_size[1], 
            #                    scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), 
            #                    interpolation=cv2.INTER_LINEAR, p=1.0),
            #回転、反転
            A.Flip(p=0.5),
            
            # A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=(-0.1,0.1),
            #                    rotate_limit=60, border_mode=0 , value=1, p=0.3),
            #                                                    #12bitaは0～1なのでvalue=1(絶対値を設定)
            #色変更
            # A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.03,
            #                           brightness_by_max=True, p=0.3),
            # A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01, p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0, hue=0, p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1)
        ])
    else:
        argument_list.extend([A.Resize(height=image_size[0], width=image_size[1], p=1.0),
                             A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1)])
    print(argument_list)
    return A.Compose(argument_list)

# def get_augmentation(image_size, train_flag=True):
#     #image_size tuple or list of [height, width]
#     # small_len = min(image_size)
#     argument_list = []
#     if train_flag:
#         argument_list.extend([
#             A.Resize(height=image_size[0], width=image_size[1], p=1.0),
#             # A.RandomResizedCrop(height=image_size[0], width=image_size[1], 
#             #                    scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), 
#             #                    interpolation=cv2.INTER_LINEAR, p=1.0),
#             #回転、反転
#             # A.Flip(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Affine(scale=(0.5, 1.5), shear=(-22.5, 22.5), rotate=(-180, 180), 
#                      translate_px = (-352 / 8, 352 / 8),
#                      interpolation=1, mask_interpolation=0, p=0.5), 
#             # A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=(-0.1,0.1),
#             #                    rotate_limit=60, border_mode=0 , value=1, p=0.3),
#             #                                                    #12bitaは0～1なのでvalue=1(絶対値を設定)
#             #色変更
#             # A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.03,
#             #                           brightness_by_max=True, p=0.3),
#             A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01, p=0.5),
#             # A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0, hue=0, p=0.5),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255)
#         ])
#     else:
#         argument_list.extend([A.Resize(height=image_size[0], width=image_size[1], p=1.0),
#                              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255)])
#     print(argument_list)
#     return A.Compose(argument_list)


def get_augmentation(image_size, train_flag=True):
    #image_size tuple or list of [height, width]
    # small_len = min(image_size)
    argument_list = []
    if train_flag:
        argument_list.extend([
            A.Resize(height=image_size[0], width=image_size[1], p=1.0),
            # A.RandomResizedCrop(height=image_size[0], width=image_size[1], 
            #                    scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), 
            #                    interpolation=cv2.INTER_LINEAR, p=1.0),
            #回転、反転
            # A.Flip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.5, 1.5), shear=(-22.5, 22.5), rotate=(-180, 180), 
                     translate_px = (-352 / 8, 352 / 8),
                     interpolation=1, mask_interpolation=0, p=0.5), 
            # A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=(-0.1,0.1),
            #                    rotate_limit=60, border_mode=0 , value=1, p=0.3),
            #                                                    #12bitaは0～1なのでvalue=1(絶対値を設定)
            #色変更
            # A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.03,
            #                           brightness_by_max=True, p=0.3),
            A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0, hue=0, p=0.5),
            # A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0, hue=0, p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255)
        ])
    else:
        argument_list.extend([A.Resize(height=image_size[0], width=image_size[1], p=1.0),
                             A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255)])
    print(argument_list)
    return A.Compose(argument_list)