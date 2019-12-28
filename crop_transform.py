from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import glob as gb
import torch
import numpy as np
import random
import warnings
from scipy import ndimage
import cv2
def rescale_crop(image, scale, num, mode):
    image_list = []
    h, w = image.size
    if mode=="train":
        trans = transforms.Compose([
        transforms.CenterCrop((int(h * scale) + 500 * scale, int(w * scale) + 500 * scale)),
        transforms.RandomCrop((int(h * scale), int(w * scale))),
        transforms.Resize((224,224)),
        transforms.RandomRotation((-10,10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    elif mode=="val":
        trans = transforms.Compose([
            transforms.CenterCrop((int(h * scale) + 500 * scale, int(w * scale) + 500 * scale)),
            transforms.RandomCrop((int(h * scale), int(w * scale))),
            transforms.Resize((224, 224)),
        ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list

def crop(image, mode):
    image_list = []
    if mode=="train":
        trans = transforms.Compose([

        transforms.RandomRotation((-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),  #change the order
    ])
    elif mode=="val":
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
    img = trans(image)
    image_list.append(img)
    return image_list

class argumentation(object):
    def __call__(self, image):
        image_list1 = rescale_crop(image, 0.2, 15, "train")
        image_list2 = rescale_crop(image, 0.4, 15, "train")
        image_list3 = rescale_crop(image, 0.6, 15, "train")
        image_list4 = rescale_crop(image, 0.8, 15, "train")
        image_list5 = crop(image, "train")
        image_list = image_list1 + image_list2 + image_list3 + image_list4 + image_list5
        nomalize = transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.7079057, 0.59156483, 0.54687315],
                             std=[0.09372108, 0.11136277, 0.12577087])])(crop) for crop in crops]))
        random.shuffle(image_list)
        image_list = nomalize(image_list)
        return image_list

class argumentation_val(object):
    def __call__(self, image):
        image_list1 = rescale_crop(image, 0.2, 2, "val")
        image_list2 = rescale_crop(image, 0.4, 2, "val")
        image_list3 = rescale_crop(image, 0.6, 2, "val")
        image_list4 = rescale_crop(image, 0.8, 2, "val")
        image_list5 = crop(image, "val")
        image_list = image_list1 + image_list2 + image_list3 + image_list4 + image_list5
        nomalize = transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.7079057, 0.59156483, 0.54687315],
                             std=[0.09372108, 0.11136277, 0.12577087])])(crop) for crop in crops]))
        image_list = nomalize(image_list)
        return image_list
