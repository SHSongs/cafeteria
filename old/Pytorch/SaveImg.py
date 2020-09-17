# pip install torchtext -U

import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import cv2

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

HumanCnt = 0

IMG_SIZE = 480
THRESHOLD = 0.45

img = cv2.imread('imgs/Cafe.jpg')
#img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width))) # 압축

trf = T.Compose([
    T.ToTensor()
])

input_img = trf(img)
X_SIZE = input_img.shape[2]
Y_SIZE = input_img.shape[1]
print(input_img.shape)

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

out = model([input_img])[0]


codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]


for box, score in zip(out['boxes'], out['scores']):
    score = score.detach().numpy()

    if score < THRESHOLD:
        continue

    box = box.detach().numpy()
    box = box.astype(int)

    # rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='b',
    #                          facecolor='none')

    # cv2.imwrite("H"+str(HumanCnt), img[box[0] : box[2] - box[0],  box[1] : box[3] - box[1]])
    crop = img[box[1] : box[3], box[0] : box[2]]
    cv2.imwrite("H" + str(HumanCnt) + ".jpg", crop)
    HumanCnt += 1





