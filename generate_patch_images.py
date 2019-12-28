import os
from tqdm import tqdm
import glob as gb
import pandas as pd
from PIL import Image
from torchvision import transforms

def get_images(root_path):
    files = []
    for ext in ['jpg']:
        files.extend(gb.glob(os.path.join(root_path, '*.{}'.format(ext))))
    return files

def rescale_crop(image, scale, num, ori=False):
    image_list = []
    h, w = image.size
    if ori:
        trans = transforms.Resize((224,224))
    else:
        trans = transforms.Compose([
        transforms.CenterCrop((int(h * scale) + 500 * scale, int(w * scale) + 500 * scale)),
        transforms.RandomCrop((int(h * scale), int(w * scale))),
        transforms.Resize((224,224))
    ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list

data_dir = "/home/wuhao/madongliang/dataset/ISIC2017/ISIC-2017_Training_Data/"
new_data_dir = "/home/wuhao/madongliang/dataset/ISIC2017/ISIC-2017_Training_Data_Patch/"
excel_dir = "/home/wuhao/madongliang/dataset/ISIC2017/ISIC-2017_Training_Part3_GroundTruth.csv"
new_excel_dir = "/home/wuhao/madongliang/dataset/ISIC2017/ISIC-2017_Training_Part3_GroundTruth_patch.csv"
images = get_images(data_dir)

if not os.path.exists(new_data_dir):
    os.makedirs(new_data_dir)


ids = []
mels = []
sks = []
for img in tqdm(images):
    image = Image.open(img)
    labels = pd.read_csv(excel_dir)
    mel = int(labels.loc[labels['image_id'] == img[img.rfind('/')+1:-4]]['melanoma'].values.squeeze())
    sk = int(labels.loc[labels['image_id'] == img[img.rfind('/')+1:-4]]['seborrheic_keratosis'].values.squeeze())
    image_list1 = rescale_crop(image, 0.2, 15)
    image_list2 = rescale_crop(image, 0.4, 15)
    image_list3 = rescale_crop(image, 0.6, 15)
    image_list4 = rescale_crop(image, 0.8, 15)
    image_list5 = rescale_crop(image, 1, 1, True)
    image_list_all = image_list1 + image_list2 + image_list3 + image_list4 + image_list5

    for i in range(len(image_list_all)):
        new_name = img[img.rfind('/')+1:-4] + '_' + str(i) + '.png'
        new_dir = os.path.join(new_data_dir, new_name)
        image_list_all[i].save(new_dir)
        labels = pd.read_csv(excel_dir)
        ids.append(new_name[:-4])
        mels.append(mel)
        sks.append(sk)
data_frame = pd.DataFrame({"image_id": ids, "melanoma": mels, "seborrheic_keratosis": sks})
data_frame.to_csv(new_excel_dir, index=False, sep=",")