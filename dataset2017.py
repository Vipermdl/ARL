from torch.utils import data
import pandas as pd
import os
import torch
from PIL import Image
import numpy as np
# from image_type import *
class ISICDataset(data.Dataset):

    def __init__(self, path, mode="training", crop=None, transform=None, task=None):
        self.path = path
        self.mode = mode
        self.samples = self.make_dataset(path)
        self.crop = crop
        self.transform = transform
        self.task = task
        self.image_list = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, melanoma, seborrheic_keratosis = self.samples[idx]
        img_name = img_path.split("/")[-1]
        image = self.pil_loader(img_path)
        if self.crop:
            image = self.crop(image)
        #hog = get_hogimage(image)
        #hog = self.totensor_nomalize(hog)
        if self.transform:
            image = self.transform(image)
        if self.task=="mel":
            return image, torch.from_numpy(np.array(int(melanoma))), img_name
        elif self.task=="sk":
            return image, torch.from_numpy(np.array(int(seborrheic_keratosis))), img_name
        else:
            return image, torch.FloatTensor([torch.from_numpy(np.array(int(melanoma))), torch.from_numpy(np.array(int(seborrheic_keratosis)))]), img_name
        

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def make_dataset(self, dir):
        images = []
        if self.mode == "training":
            img_dir = os.path.join(dir, "ISIC-2017_Training_Data_Patch")
            csv_filename = os.path.join(dir, "ISIC-2017_Training_Part3_GroundTruth_patch.csv")
        if self.mode == "validation":
            img_dir = os.path.join(dir, "ISIC-2017_Validation_Data")
            csv_filename = os.path.join(dir, "ISIC-2017_Validation_Part3_GroundTruth.csv")
        if self.mode == "testing":
            img_dir = os.path.join(dir, "ISIC-2017_Test_v2_Data")
            csv_filename = os.path.join(dir, "ISIC-2017_Test_v2_Part3_GroundTruth.csv")
        label_list = pd.read_csv(csv_filename)

        for index, row in label_list.iterrows():
            if self.mode == "training":
                images.append((os.path.join(img_dir, row["image_id"] + ".png"), row["melanoma"], row["seborrheic_keratosis"]))
            else:
                images.append((os.path.join(img_dir, row["image_id"] + ".jpg"), row["melanoma"], row["seborrheic_keratosis"]))
        return images
