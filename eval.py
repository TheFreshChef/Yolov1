import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import YOLOv1
from src.utils import non_max_suppression

from torchmetrics.detection.mean_ap import MeanAveragePrecision

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE   = 448
BATCH_SIZE = 8
CONF_THRESH = 0.05
IOU_THRESH  = 0.5

CHECKPOINTS = [
    'models/yolov1_epoch050.pth',
    'models/yolov1_epoch100.pth',
    'models/yolov1_epoch150.pth',
    'models/yolov1_final.pth',
]

VOC_ROOT = 'data/VOCdevkit/VOC2007'

def load_ground_truth(xml_path):
    tree = ET.parse(xml_path)
    boxes, labels = [], []
    for obj in tree.findall('object'):
        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)
        cls_name = obj.find('name').text.lower().strip()
        cls_id = __import__('src.dataset', fromlist=['VOC_CLASSES']).VOC_CLASSES[cls_name]
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls_id)
    return {
        'boxes':  torch.tensor(boxes, dtype=torch.float32, device=DEVICE),
        'labels': torch.tensor(labels, dtype=torch.int64,   device=DEVICE),
    }

def make_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    from src.dataset import VOCDataset
    ds = VOCDataset(
        root='data/VOCdevkit',
        year='2007',
        image_set='trainval',
        transform=transform,
        S=7, B=2, C=20,
        img_size=IMG_SIZE
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return ds, loader

def evaluate(ckpt_path, ds, loader):
    # load model
    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    metric = MeanAveragePrecision(iou_thresholds=[IOU_THRESH])

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            preds = model(imgs)

            for i in range(imgs.size(0)):
                boxes, scores, cls_ids = non_max_suppression(
                    preds[i].cpu(), conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH
                )

                metric.update(
                    [ {
                        'boxes':  torch.tensor(boxes,  dtype=torch.float32, device=DEVICE),
                        'scores': torch.tensor(scores, dtype=torch.float32, device=DEVICE),
                        'labels': torch.tensor(cls_ids, dtype=torch.int64, device=DEVICE)
                    } ],
                    [ load_ground_truth(
                        ds.ann_paths[batch_idx * BATCH_SIZE + i]
                      )
                    ]
                )

    stats = metric.compute()
    return stats['map_50'].item()

if __name__ == '__main__':
    ds, loader = make_test_loader()
    for ckpt in CHECKPOINTS:
        mAP50 = evaluate(ckpt, ds, loader) * 100
        print(f"{os.path.basename(ckpt):>20} → mAP@0.5 = {mAP50:5.2f}%")