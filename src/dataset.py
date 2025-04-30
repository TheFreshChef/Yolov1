import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# 20 classes
VOC_CLASSES = {cls: i for i, cls in enumerate([
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor',
])}

def encode_target(boxes, labels, S=7, B=2, C=20, img_size=448):
    """
    boxes: (N,4) in pixel coords on a img_size×img_size image
    labels: (N,) integer class ids
    returns: (S, S, B*5 + C) tensor
    """
    target = np.zeros((S, S, B*5 + C), dtype=np.float32)

    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2 = box
        xc = ((x1 + x2) / 2) / img_size
        yc = ((y1 + y2) / 2) / img_size
        w  =  (x2 - x1)        / img_size
        h  =  (y2 - y1)        / img_size

        xc = min(max(xc, 0.0), 1 - 1e-6)
        yc = min(max(yc, 0.0), 1 - 1e-6)

        i = int(xc * S)
        j = int(yc * S)
        x_cell = xc * S - i
        y_cell = yc * S - j

        for b in range(B):
            target[j, i, b*5 + 0:b*5 + 5] = [x_cell, y_cell, w, h, 1.0]
        target[j, i, B*5 + cls] = 1.0

    return torch.from_numpy(target)

class VOCDataset(Dataset):
    def __init__(self,
                 root='data/VOCdevkit',
                 year='2007',
                 image_set='train',
                 transform=None,
                 S=7, B=2, C=20,
                 img_size=448):
        voc_root = os.path.join(root, f'VOC{year}')
        split_f  = os.path.join(voc_root, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(split_f) as f:
            ids = [x.strip() for x in f]

        self.img_paths = [os.path.join(voc_root, 'JPEGImages', f"{i}.jpg") for i in ids]
        self.ann_paths = [os.path.join(voc_root, 'Annotations', f"{i}.xml") for i in ids]
        self.transform = transform
        self.S, self.B, self.C = S, B, C
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        orig_w, orig_h = img.size

        tree = ET.parse(self.ann_paths[idx])
        boxes, labels = [], []
        for obj in tree.findall('object'):
            cls_name = obj.find('name').text.lower().strip()
            labels.append(VOC_CLASSES[cls_name])
            bb = obj.find('bndbox')
            xmin = float(bb.find('xmin').text)
            ymin = float(bb.find('ymin').text)
            xmax = float(bb.find('xmax').text)
            ymax = float(bb.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes  = np.array(boxes,  dtype=np.float32)  
        labels = np.array(labels, dtype=np.int64)

        if self.transform:
            img = self.transform(img)

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        boxes[:, [0,2]] *= scale_x  
        boxes[:, [1,3]] *= scale_y 

        target = encode_target(
            boxes, labels,
            S=self.S, B=self.B, C=self.C,
            img_size=self.img_size
        )

        return img, target