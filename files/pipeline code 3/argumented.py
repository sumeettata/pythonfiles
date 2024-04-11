import torch
from torchvision.transforms import v2
import glob
import os
from PIL import Image

path = "D:/Work/Tata Steel/roller detection/argumented"
files = glob.glob(path+'/*')


def apply_transform(image,no):
    transforms = v2.RandomApply([
        v2.RandomHorizontalFlip(p=no*0.1),
        v2.RandomVerticalFlip(p=no*0.1),
        v2.RandomRotation(degrees=random.choice(range)),
        v2.RandomAffine(degrees=no*36),
        v2.RandomPerspective(),
        v2.ElasticTransform(),
        v2.ColorJitter(),
        v2.RandomPhotometricDistort(),
        v2.RandomGrayscale(),
        v2.GaussianBlur(kernel_size=3),
        v2.RandomInvert(),
        v2.RandomAdjustSharpness(sharpness_factor=3),
        v2.RandomAutocontrast(),
        v2.RandomEqualize()
    ])


i=0
j=0
while i < 50:
    i = i+1
    for file in files:
        j=j+1
        img = Image.open(file)
        img = transforms(img)
        img.save(os.path.dirname(file)+'/argumented_'+str(i)+str(j)+'_'+os.path.basename(file))