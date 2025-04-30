import argparse
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from src.model import YOLOv1
from src.utils import non_max_suppression

parser = argparse.ArgumentParser()
parser.add_argument('--image','-i',required=True)
parser.add_argument('--output','-o','--out',default='result.jpg')
args = parser.parse_args()

model = YOLOv1(7,2,20)
state = torch.load('models/yolov1_final.pth', map_location='cpu')
print(f"[+] Loaded checkpoint with {len(state)} params.")
model.load_state_dict(state)
model.eval()

transform = transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()])
img = Image.open(args.image).convert('RGB')
inp = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(inp)

# try a lower threshold first
conf_thresh = 0.05
boxes, scores, classes = non_max_suppression(
    out.squeeze(0), conf_thresh=conf_thresh, iou_thresh=0.5
)

print(f"[+] After NMS (conf>{conf_thresh}), kept {len(boxes)} boxes")
for b,s,c in zip(boxes, scores, classes):
    print(f"    Box {b} | score {s:.3f} | class {c}")

draw = ImageDraw.Draw(img)
for (x1,y1,x2,y2), cls in zip(boxes, classes):
    draw.rectangle([x1,y1,x2,y2], width=2)
    draw.text((x1,y1), str(cls), fill='white')

img.save(args.output)
print(f"[+] Saved result to {args.output}")
