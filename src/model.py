import torch
import torch.nn as nn

def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class YOLOv1(nn.Module):
    """
    From-scratch YOLO-v1 (24 conv layers + 2 FC layers) as in the original paper.
    S: grid size, B: boxes per cell, C: number of classes.
    """
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        self.conv = nn.Sequential(
            # (224×224) → (112×112)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),

            # (112×112) → (56×56)
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),

            # (56×56) conv stack
            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),

            # (28×28) ×4 of {1×1,3×3} with 512 filters
            *sum([[  
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
            ] for _ in range(4)], []),
            # reduce then expand
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),  # → (14×14)

            # (14×14) ×2 of {1×1,3×3} with 1024 filters
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2,2),  # 14 → 7
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            # ↓↓↓ now from 1024×7×7, not 14×14
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (5 * B + C))
        )

        self.apply(_init_weights)

    def forward(self, x):
        # x: (batch, 3, 448, 448)
        x = self.conv(x)          # → (batch, 1024, 7, 7)
        x = self.fc(x)            # → (batch, S*S*(5B + C))
        return x.view(-1, self.S, self.S, 5 * self.B + self.C)