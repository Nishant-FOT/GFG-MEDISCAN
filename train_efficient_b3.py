import os
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --------------------
# GPU OPTIMIZATIONS
# --------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------
# CONFIG (RTX 4050 SAFE)
# --------------------
DATA_DIR = "data/combined"
IMG_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 40
LR = 3e-4
NUM_WORKERS = 4
GRAD_ACCUM_STEPS = 2   # Effective batch = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# --------------------
# DATASET
# --------------------
class SkinDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMG_DIR, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label

# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print("Classes:", classes)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # --------------------
    # TRANSFORMS (DERM SAFE)
    # --------------------
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = SkinDataset(train_df, class_to_idx, train_tfms)
    val_dataset = SkinDataset(val_df, class_to_idx, val_tfms)

    # --------------------
    # CLASS BALANCING
    # --------------------
    train_labels = [class_to_idx[l] for l in train_df["label"]]
    counts = Counter(train_labels)

    class_weights = torch.tensor(
        [1.0 / counts[i] for i in range(len(classes))],
        dtype=torch.float
    ).to(DEVICE)

    sampler = WeightedRandomSampler(
        [class_weights[l].item() for l in train_labels],
        num_samples=len(train_labels),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --------------------
    # MODEL
    # --------------------
    model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )
    model.to(DEVICE)

    # --------------------
    # LOSS & OPTIM
    # --------------------
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # --------------------
    # TRAIN LOOP
    # --------------------
    best_acc = 0.0
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        model.train()
        correct = 0

        for step, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dataset)

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes
                },
                "efficientnet_b3_skin_best.pth"
            )
            print("✅ Best model saved")

        scheduler.step()

    print("🎉 Training complete")
