# train.py
import os
import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import VOCDataset
from src.model   import YOLOv1
from src.loss    import YOLOLoss

def main():
    print("[*] Starting training…")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    train_ds = VOCDataset(
        root='data/VOCdevkit',
        year='2007',
        image_set='trainval', 
        transform=transform,
        S=7, B=2, C=20,
        img_size=448
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    print(f"[*] Loaded {len(train_ds)} images into DataLoader")

    model     = YOLOv1(S=7, B=2, C=20).to(device)
    criterion = YOLOLoss(S=7, B=2, C=20)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,             
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200           
    )

    os.makedirs('models', exist_ok=True)
    for epoch in range(1, 201):
        epoch_start_time = time.time()
        print(f"---- Epoch {epoch} ----")
        model.train()
        total_loss = 0.0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            preds = model(imgs)              
            loss  = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | loss: {avg_loss:.4f} | lr: {lr:.1e} | time: {epoch_duration:.2f}s")

        if epoch % 50 == 0:
            ckpt = f"models/yolov1_epoch{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}")

    # final save
    torch.save(model.state_dict(), 'models/yolov1_final.pth')
    print("Training complete. Final model saved to models/yolov1_final.pth")

if __name__ == "__main__":
    main()
