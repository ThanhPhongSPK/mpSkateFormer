import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import math
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime

# Import modules
sys.path.insert(0, "/home/ducduy/Phong")

from my_src.utils.dataloader import MediaPipeSkateDataset, NTU60_CLASSES, _get_ntu60_split
from my_src.models.SkateFormer import create_mediapipe_skateformer
from my_src.utils.metrics import Metrics

# --- Configurations ---
BASE_DATA_PATH = '/home/ducduy/Phong/mp_skeletons'

# Folder containing data
DATA_FOLDERS = [
    'nturgbd_rgb_s001_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s002_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s003_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s004_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s005_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s006_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s007_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s008_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s009_single_actor/nturgb+d_skeleton',
    'nturgbd_rgb_s010_single_actor/nturgb+d_skeleton',
]

result_folder = f"my_src/results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(result_folder, exist_ok=True)

# Resume training if needed
RESUME_PATH = input("Enter checkpoint path to resume (or leave blank to train from scratch): ").strip()
START_EPOCH_INPUT = input("Enter start epoch (default 1): ").strip()
START_EPOCH = int(START_EPOCH_INPUT) if START_EPOCH_INPUT else 1

# If set BATCH_SIZE = 64 and BASE_LR = 1e-3, use small ACCUMULATION_STEPS 
BATCH_SIZE = 64
BASE_LR = 1e-3
ACCUMULATION_STEPS = 2

# BASE_LR = 2e-4
# BATCH_SIZE = 16
# ACCUMULATION_STEPS = 4

EPOCHS = 300
WARMUP_EPOCHS = 25
NUM_CLASSES = 60
FRAMES_LEN = 64

WEIGHT_DECAY = 0.1

SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, criterion, optimizer, scaler, mixup_fn, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")

    # Reset gradients before loop
    optimizer.zero_grad()

    for i, (data, targets, index_t) in enumerate(loop):
        data, targets, index_t = data.to(device), targets.to(device), index_t.to(device)

        if mixup_fn is not None:
            data, targets = mixup_fn(data, targets) # one hot vector

        # Forward
        with autocast(device_type='cuda'):
            outputs = model(data, index_t)
            loss = criterion(outputs, targets)
            loss = loss / ACCUMULATION_STEPS

        # --- SAFETY CHECK ---
        if torch.isnan(loss):
            print(f"\n[WARNING] NaN loss detected at Epoch {epoch}, Batch {i}. Skipping step.")
            continue
        # --------------------

        scaler.scale(loss).backward()

        # Update weights every ACCUMULATION_STEPS
        if (i + 1) % ACCUMULATION_STEPS == 0:
            # Unscale before clip grad
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # --- SAFETY CHECK BACKWARD---
            if torch.isnan(grad_norm):
                print(f"\n[WARNING] NaN gradient detected at Epoch {epoch}, Batch {i}. Skipping update.")
                optimizer.zero_grad()
                scaler.step(optimizer)
                scaler.update()
                continue
            
            else:
                # Backward
                scaler.step(optimizer)
                scaler.update()
            # ---------------------

            # Reset gradients
            optimizer.zero_grad()

        # Stats
        if not torch.isnan(loss):
            running_loss += loss.item() * ACCUMULATION_STEPS

        _, predicted = outputs.max(1)
        _, target_indices = targets.max(1) # take the biggest class weight
        
        total += targets.size(0)
        correct += predicted.eq(target_indices).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    return running_loss / len(loader), 100.*correct / total

def validate(model, loader, criterion, device, epoch, metrics_collector=None):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data, targets, index_t in loader:
            data, targets, index_t = data.to(device), targets.to(device), index_t.to(device)
            outputs = model(data, index_t)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Collect predictions for metrics
            if metrics_collector is not None:
                metrics_collector.collect(outputs, targets)

        acc = 100. * correct / total
        val_loss = running_loss / len(loader)
        print(f"Epoch {epoch} [Val] Loss: {running_loss/len(loader):.4f} | Acc: {acc:.2f}%")
        return acc, val_loss
    
# Custom Cosine Scheduler with Warmup
def get_scheduler(optimizer, num_epochs, warmup_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full_data_folders = [os.path.join(BASE_DATA_PATH, f) for f in DATA_FOLDERS]

    print("--> Splitting data using NTU X-Sub protocol...")
    train_files, val_files = _get_ntu60_split(full_data_folders, split_type='xsub')
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total folders: {len(DATA_FOLDERS)}")
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    print(f"========================\n")

    train_set = MediaPipeSkateDataset(train_files, mode='train', frames_len=FRAMES_LEN)
    val_set = MediaPipeSkateDataset(val_files, mode='val', frames_len=FRAMES_LEN)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Init model and metrics
    model = create_mediapipe_skateformer(num_classes=NUM_CLASSES).to(device)

    # Initialize metrics collector
    metrics_collector = Metrics(result_folder, num_classes=NUM_CLASSES, class_names=NTU60_CLASSES)

    # If resume from checkpoint
    current_best_acc = 0.0
    start_epoch = START_EPOCH
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"--> Loading checkpoint: {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location=device)

        if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
             model.load_state_dict(checkpoint)
        else:
             model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

        print("--> Resume weights success!")

        print("--> Validating loaded model...")
        current_best_acc, _ = validate(model, val_loader, nn.CrossEntropyLoss(), device, epoch=0)
        print(f"--> Current Best Acc: {current_best_acc:.2f}%")

    else:
        print("--> No valid checkpoint found or RESUME_PATH is None. Training from scratch.")

    # Mixup Object
    mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, 
    switch_prob=0.5, mode='batch',
    label_smoothing=0.1, num_classes=NUM_CLASSES
    )

    # Optimizer & Loss 
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

    # Scaler cho AMP
    scaler = GradScaler('cuda')

    # Loop
    best_acc = current_best_acc
    
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(start_epoch, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, train_criterion, optimizer, scaler, mixup_fn, device, epoch)
        val_acc, val_loss = validate(model, val_loader, val_criterion, device, epoch, metrics_collector=metrics_collector)
        
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch}| LR: {curr_lr:.2e} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save epoch metrics
        metrics_collector.save_epoch_metrics(epoch, train_loss, val_loss, train_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{result_folder}/best_skateformer_model.pth")
            print(f"----> Saved Best Model: {best_acc:.2f}%")

    validate(model, val_loader, criterion, device, epoch=EPOCHS, metrics_collector=metrics_collector)
    metrics_collector.save_confusion_matrix()

if __name__ == "__main__":
    main()