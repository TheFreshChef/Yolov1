import torch

def iou(box1, box2):
    """
    box1, box2: (...,4) in (x1,y1,x2,y2) format
    returns IoU map of shape (...)
    """
    inter_x1 = torch.max(box1[...,0], box2[...,0])
    inter_y1 = torch.max(box1[...,1], box2[...,1])
    inter_x2 = torch.min(box1[...,2], box2[...,2])
    inter_y2 = torch.min(box1[...,3], box2[...,3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1[...,2] - box1[...,0]) * (box1[...,3] - box1[...,1])
    area2 = (box2[...,2] - box2[...,0]) * (box2[...,3] - box2[...,1])
    union = area1 + area2 - inter_area + 1e-6

    return inter_area / union

def non_max_suppression(pred_tensor, conf_thresh=0.25, iou_thresh=0.5,
                        S=7, B=2, C=20, img_size=448):
    """
    pred_tensor: (S, S, B*5 + C) output for one image
    returns lists: boxes [(x1,y1,x2,y2)...], scores [...], class_ids [...]
    """
    # 1) split into box preds and class probs
    # pred_tensor[..., :5*B] -> (S, S, B*5)
    # pred_tensor[..., 5*B:] -> (S, S, C)
    bboxes = pred_tensor[..., :5*B] \
                .view(S, S, B, 5)      # (S, S, B, 5)
    class_probs = pred_tensor[..., 5*B:]  # (S, S, C)

    final_boxes = []
    final_scores = []
    final_cls_ids = []

    # 2) for each grid cell and each box
    for i in range(S):
        for j in range(S):
            # per‐cell class probabilities
            cell_cls = class_probs[i, j]                  # (C,)
            cls_conf, cls_id = torch.max(cell_cls, dim=0) # scalars

            for b in range(B):
                x, y, w, h, conf = bboxes[i, j, b]

                # combined score = objectness * class confidence
                score = conf * cls_conf
                if score < conf_thresh:
                    continue

                # convert to absolute pixel coords on img_size×img_size
                cx = (j + x) * (img_size / S)
                cy = (i + y) * (img_size / S)
                bw = abs(w) * img_size
                bh = abs(h) * img_size

                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

                final_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                final_scores.append(score.item())
                final_cls_ids.append(cls_id.item())

    keep = []
    idxs = sorted(range(len(final_scores)),
                  key=lambda k: final_scores[k],
                  reverse=True)

    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [
            idx for idx in idxs
            if iou(
                torch.tensor(final_boxes[current]),
                torch.tensor(final_boxes[idx])
            ) < iou_thresh
        ]

    return (
        [ final_boxes[i] for i in keep ],
        [ final_scores[i] for i in keep ],
        [ final_cls_ids[i] for i in keep ]
    )