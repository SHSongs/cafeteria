# pip install torchtext -U

import torch
import torchvision
from torchvision import models
import torchvision.transforms as T


import cv2

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

HumanCnt = 0

IMG_SIZE = 480
THRESHOLD = 0.45

img = cv2.imread('imgs/Cafe.jpg')

trf = T.Compose([
    T.ToTensor()
])

input_img = trf(img)
print(input_img.shape)

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
out = model([input_img])[0]



for box, score in zip(out['boxes'], out['scores']):
    score = score.detach().numpy()

    if score < THRESHOLD:
        continue

    box = box.detach().numpy()
    box = box.astype(int)

    crop = img[box[1] : box[3], box[0] : box[2]]
    cv2.imwrite("H" + str(HumanCnt) + ".jpg", crop)
    HumanCnt += 1





