import torch
import torch.nn as nn
from src.utils import iou

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20,
                 lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        """
        pred:   (N, S, S, B*5 + C)  raw network output
        target: (N, S, S, B*5 + C)  encoded ground truth
        """
        N = pred.size(0)
        device = pred.device

        pred_boxes = pred[..., :5*self.B] \
                         .view(N, self.S, self.S, self.B, 5)
        
        pred_cls   = pred[..., 5*self.B:]


        true_boxes = target[..., :5] \
                         .unsqueeze(3) \
                         .expand(N, self.S, self.S, self.B, 5)
        
        true_conf  = target[..., 4] \
                         .unsqueeze(3) \
                         .expand(N, self.S, self.S, self.B)
        
        true_cls   = target[..., 5*self.B:]

        obj_mask = (target[..., 4] > 0)  

        ious = iou(pred_boxes[..., :4], true_boxes[..., :4]) 
        # Pick the box with highest IoU in each cell
        best_box = torch.argmax(ious, dim=-1)  

        # Build a boolean mask for the “responsible” box per cell
        best_box_mask = torch.zeros_like(ious, dtype=torch.bool)  
        best_box_mask.scatter_(-1, best_box.unsqueeze(-1), True)

        coord_mask = best_box_mask & obj_mask.unsqueeze(-1)       
        # expand to cover the 4 coord channels
        coord_mask_box = coord_mask.unsqueeze(-1).expand(-1,-1,-1,-1,4)  

        pred_box_coords = pred_boxes[..., :4]
        true_box_coords = true_boxes[..., :4]

        if coord_mask_box.sum() > 0:
            coord_loss = self.mse(
                pred_box_coords[coord_mask_box],
                true_box_coords[coord_mask_box]
            )
        else:
            coord_loss = torch.tensor(0., device=device)

        pred_conf = pred_boxes[..., 4]  # (N,S,S,B)

        if coord_mask.sum() > 0:
            obj_loss = self.mse(
                pred_conf[coord_mask],
                true_conf[coord_mask]
            )
        else:
            obj_loss = torch.tensor(0., device=device)

        noobj_mask = (~obj_mask).unsqueeze(-1) \
                          .expand(N, self.S, self.S, self.B)   # (N,S,S,B)
        noobj_loss = self.mse(
            pred_conf[noobj_mask],
            torch.zeros_like(pred_conf[noobj_mask], device=device)
        )

        class_mask = obj_mask.unsqueeze(-1).expand(-1,-1,-1,self.C)  # (N,S,S,C)
        if class_mask.sum() > 0:
            class_loss = self.mse(
                pred_cls[class_mask],
                true_cls[class_mask]
            )
        else:
            class_loss = torch.tensor(0., device=device)

        loss = (
            self.lambda_coord * coord_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        ) / N

        return loss
