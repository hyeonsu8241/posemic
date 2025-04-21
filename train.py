import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
# from ainet import AttentionImitationNetwork, PoseDataset
from ainet_wo_attn import AttentionImitationNetwork, PoseDataset
from util_train_2 import path_load
from torch.nn.utils.rnn import pad_sequence
from losses import FocalLoss, LDAMLoss
import mediapipe as mp

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MEDIAPIPE_DISABLE_GPU'] = "0"
os.environ['TFLITE_ENABLE_GPU_DELEGATE'] = "1"
os.environ['TFLITE_ENABLE_XNNPACK'] = "0"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

def custom_collate_fn(batch):
    child_data, teach_data, labels = zip(*batch)

    child_data = [torch.tensor(item, dtype=torch.float32).to("cuda") for item in child_data]
    teach_data = [torch.tensor(item, dtype=torch.float32).to("cuda") for item in teach_data]
    labels = torch.tensor(labels, dtype=torch.float32).to("cuda")

    padded_child_frames = pad_sequence(child_data, batch_first=True)
    padded_teacher_frames = pad_sequence(teach_data, batch_first=True)

    return padded_child_frames.to("cuda"), padded_teacher_frames.to("cuda"), labels.to("cuda")

max_iter = 50
initial_lr = 0.01
k_folds = 3
batch_size = 32
device = 'cuda:0'

vid_paths, teach_paths, gts = path_load(
    "/dev/hdd/hs/X-Pose/data_paths.txt",
    "/dev/hdd/hs/X-Pose/gts_ib_busan.txt"
)

kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(vid_paths, gts)):
    if fold == 0:
        continue
    print(f"Fold {fold+1}/{k_folds} Start...")

    train_dataset = PoseDataset(
        [vid_paths[i] for i in train_idx],
        [teach_paths[i] for i in train_idx],
        [gts[i] for i in train_idx]
    )

    val_dataset = PoseDataset(
        [vid_paths[i] for i in val_idx],
        [teach_paths[i] for i in val_idx],
        [gts[i] for i in val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True,
    #     collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
    # )
    # val_loader = DataLoader(
    #     val_dataset, batch_size=1, shuffle=False,
    #     collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
    # )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=2,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

    model = AttentionImitationNetwork(pose).to("cuda")
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss()
    # criterion = LDAMLoss()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, max_iter + 1):
        model.train()
        total_loss = 0

        for child_batch, teacher_batch, labels in train_loader:
            # print(child_path)
            # print("batch shape")
            # print(child_batch.shape, teacher_batch.shape)
            child_batch, teacher_batch, labels = child_batch.to("cuda"), teacher_batch.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()
            outputs = model(child_batch, teacher_batch)
            outputs = outputs.transpose(-2, -1)
            # outputs = outputs.squeeze(0)
            labels = labels.unsqueeze(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch}/{max_iter}, Loss: {total_loss / len(train_loader):.4f}')

        if epoch % 5 == 0:
            model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for child_batch, teacher_batch, labels in val_loader:
                    child_batch, teacher_batch, labels = child_batch.to("cuda"), teacher_batch.to("cuda"), labels.to("cuda")
                    output = torch.sigmoid(model(child_batch, teacher_batch)).squeeze()
                    preds = (output > 0.5).float()

                    all_preds.append(preds.cpu().numpy().tolist())
                    all_labels.append(labels.cpu().numpy().tolist())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            cm = confusion_matrix(all_labels, all_preds)

            print(f'[Fold {fold+1}/{k_folds}] Epoch {epoch}: Accuracy: {acc:.4f}, F1 Score: {f1:.4f}')
            print(f'Confusion Matrix:\n{cm}')

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                os.makedirs(f"./checkpoints/adamw/fold_{fold+1}", exist_ok=True)
                torch.save(model.state_dict(), f"./checkpoints/adamw/fold_{fold+1}/best_model.pt")

    print(f"Fold {fold+1} Complete! Best Accuracy: {best_acc:.2f} at Epoch {best_epoch}")
