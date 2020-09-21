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

HumanList = []

IMG_SIZE = 480
THRESHOLD = 0.30

img = Image.open('imgs/Cafe.jpg')
img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width))) # 압축

plt.figure(figsize=(16, 16))

plt.imshow(img)

trf = T.Compose([
    T.ToTensor()
])

input_img = trf(img)

print(input_img.shape)

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

out = model([input_img])[0]

print(out.keys())

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]

fig, ax = plt.subplots(1, figsize=(16, 16))


for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
    score = score.detach().numpy()

    if score < THRESHOLD:
        continue

    box = box.detach().numpy()
    keypoints = keypoints.detach().numpy()[:, :2]

    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='b',
                             facecolor='none')

    ax.add_patch(rect)

    # # 17 keypoints
    # for i, k in enumerate(keypoints):
    #     circle = patches.Circle((k[0], k[1]), radius=10, facecolor='r')
    #     ax.add_patch(circle)
    #     print(k)
    #     if i > 3:
    #         break

    # # draw path
    # # left arm
    # path = Path(keypoints[5:10:2], codes)
    # line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    # ax.add_patch(line)
    #
    # # right arm
    # path = Path(keypoints[6:11:2], codes)
    # line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    # ax.add_patch(line)
    #
    # # left leg
    # path = Path(keypoints[11:16:2], codes)
    # line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    # ax.add_patch(line)
    #
    # # right leg
    # path = Path(keypoints[12:17:2], codes)
    # line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
    # ax.add_patch(line)

ax.imshow(img.numpy())

plt.show()
