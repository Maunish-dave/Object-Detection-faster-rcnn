import pandas as pd
import numpy as np
import cv2
import os
import re
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, Dataset
from matplotlib import pyplot as plt
import warnings
from ObjDecWeb import settings

warnings.filterwarnings("ignore")

classes = {
    0: u"__background__",
    1: u"person",
    2: u"bicycle",
    3: u"car",
    4: u"motorcycle",
    5: u"airplane",
    6: u"bus",
    7: u"train",
    8: u"truck",
    9: u"boat",
    10: u"traffic light",
    11: u"fire hydrant",
    12: u"stop sign",
    13: u"parking meter",
    14: u"bench",
    15: u"bird",
    16: u"cat",
    17: u"dog",
    18: u"horse",
    19: u"sheep",
    20: u"cow",
    21: u"elephant",
    22: u"bear",
    23: u"zebra",
    24: u"giraffe",
    25: u"backpack",
    26: u"umbrella",
    27: u"handbag",
    28: u"tie",
    29: u"suitcase",
    30: u"frisbee",
    31: u"skis",
    32: u"snowboard",
    33: u"sports ball",
    34: u"kite",
    35: u"baseball bat",
    36: u"baseball glove",
    37: u"skateboard",
    38: u"surfboard",
    39: u"tennis racket",
    40: u"bottle",
    41: u"wine glass",
    42: u"cup",
    43: u"fork",
    44: u"knife",
    45: u"spoon",
    46: u"bowl",
    47: u"banana",
    48: u"apple",
    49: u"sandwich",
    50: u"orange",
    51: u"broccoli",
    52: u"carrot",
    53: u"hot dog",
    54: u"pizza",
    55: u"donut",
    56: u"cake",
    57: u"chair",
    58: u"couch",
    59: u"potted plant",
    60: u"bed",
    61: u"dining table",
    62: u"toilet",
    63: u"tv",
    64: u"laptop",
    65: u"mouse",
    66: u"remote",
    67: u"keyboard",
    68: u"cell phone",
    69: u"microwave",
    70: u"oven",
    71: u"toaster",
    72: u"sink",
    73: u"refrigerator",
    74: u"book",
    75: u"clock",
    76: u"vase",
    77: u"scissors",
    78: u"teddy bear",
    79: u"hair drier",
    80: u"toothbrush",
    81: u"object",
    82: u"object",
    83: u"object",
    84: u"object",
    85: u"object",
    86: u"object",
    87: u"object",
    88: u"object ",
    89: u"object",
    90: u"object",
    91: u"object",
}


def detection(path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False
    )
    load_path = settings.MEDIA_ROOT + "/faste_rcnn50_coco.pth"
    model.load_state_dict(torch.load(load_path))
    device = torch.device("cpu")
    model.eval()
    x = model.to(device)
    image = Image.open(path)

    imsize = 256
    tfms = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = tfms(image).float()
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)

    def save_image(img, bboxes, path, labels):
        img = Image.fromarray(np.uint8(img * 255))
        img1 = ImageDraw.Draw(img)
        for b, l in zip(bboxes, labels):
            w = b[2]
            h = b[3]
            img1.rectangle([(b[0], b[1]), (w, h)], fill=None, outline="red")
            img1.text((b[0], b[1] - 10), classes[l], fill=(0, 255, 0))
        img.save(settings.MEDIA_ROOT + "/images/abc.jpg", "JPEG")
        return None

    image = image[0].permute(1, 2, 0).cpu().numpy()
    thres = output[0]["scores"].detach().cpu().numpy() > 0.7
    labels = output[0]["labels"].detach().cpu().numpy()
    labels = labels[thres]
    boxes = output[0]["boxes"].detach().cpu().numpy()[thres]
    print(boxes)
    save_image(image, boxes, path, labels)

