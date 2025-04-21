import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
import mediapipe as mp
import re
# from mp_model_res_mini_2 import ResNet
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import tensorflow.lite as tflite
from tflite_runtime.interpreter import load_delegate
import torchvision.transforms as transforms

# gpu_delegate = load_delegate("libtensorflowlite_gpu_delegate.so")
# gpu_delegate = load_delegate("libdelegate.so")

# interpreter = tflite.Interpreter(
#     model_path="model.tflite",
#     experimental_delegates=[gpu_delegate]
# )
#
# interpreter.allocate_tensors()

def similarity_map_to_rgb_tensor_no_plt(similarity_map, resize_size=(256, 256)):
    # print(similarity_map.shape)
    normalized_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())

    def viridis_color(value):
        c1 = torch.tensor([0.267004, 0.004874, 0.329415])
        c2 = torch.tensor([0.232325, 0.351986, 0.590159])
        c3 = torch.tensor([0.098415, 0.618603, 0.840995])
        c4 = torch.tensor([0.562095, 0.887594, 0.640152])
        c5 = torch.tensor([0.993248, 0.906157, 0.143936])

        if value < 0.25:
            return c1 + (c2 - c1) * (value / 0.25)
        elif value < 0.5:
            return c2 + (c3 - c2) * ((value - 0.25) / 0.25)
        elif value < 0.75:
            return c3 + (c4 - c3) * ((value - 0.5) / 0.25)
        else:
            return c4 + (c5 - c4) * ((value - 0.75) / 0.25)

    # 각 픽셀에 컬러맵 적용
    rgb_map = torch.stack([viridis_color(v) for v in normalized_map.flatten()]).reshape(*normalized_map.shape, 3)
    # print(rgb_map.shape)
    rgb_mapp = torch.zeros((rgb_map.shape[0], 3, rgb_map.shape[1], rgb_map.shape[2]), dtype=torch.float32)

    # 채널 순서 변경 (HWC -> CHW)
    for it in range(rgb_map.shape[0]):
         rgb_mapp[it, :, :, :] = rgb_map[it, :, :, :].permute(2, 0, 1).float()

    rgb_tensor = rgb_mapp
    # 이미지 크기 조정
    resize_transform = transforms.Resize(resize_size)
    resized_tensor = resize_transform(rgb_tensor)

    return resized_tensor.to("cuda")

def apply_colormap(tensor):
    batch_size = tensor.size(0)
    colormap_applied = []

    for i in range(batch_size):
        img = tensor[i].cpu().numpy()  # (Frame, Frame)

        # Normalize for colormap (0~1 범위로 스케일)
        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        # Apply colormap
        cmap = plt.get_cmap('viridis')
        img_colormap = cmap(img_normalized)[:, :, :3]  # RGBA 중 RGB만 사용

        # Tensor로 변환 (C, H, W 형태로)
        img_colormap = torch.tensor(img_colormap).permute(2, 0, 1).float()
        colormap_applied.append(img_colormap)

    return torch.stack(colormap_applied).cuda()  # 배치로 변환하여 반환

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)

def resize_image(image, target_size=(1920, 1080)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def get_closest_person_keypoints(image):
    h, w, _ = image.shape
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # img_high = resize_image(image_rgb)

    results = pose.process(image)
    # results = pose.process(cv2.cvtColor(img_high, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return torch.zeros((33, 2), dtype=torch.float32, device="cuda")

    keypoints_list = []
    distances = []

    for landmarks in [results.pose_landmarks]:
        keypoints = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])

        x_min, y_min = np.min(keypoints, axis=0)
        x_max, y_max = np.max(keypoints, axis=0)

        bbox_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

        image_center = np.array([w / 2, h / 2])

        distance = np.linalg.norm(bbox_center - image_center)

        keypoints_list.append(keypoints)
        distances.append(distance)

    min_idx = np.argmin(distances)
    closest_keypoints = keypoints_list[min_idx]

    return torch.tensor(closest_keypoints, dtype=torch.float32, device="cuda")


class PoseDataset(Dataset):
    def __init__(self, child_paths, teacher_paths, labels):
        self.child_paths = child_paths
        self.teacher_paths = teacher_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        child_dir = self.child_paths[idx]
        teacher_dir = self.teacher_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 기존 전처리 적용
        child_frame = self.process_frames_single(child_dir)
        teacher_frame = self.process_frames_single(teacher_dir)

        return child_frame, teacher_frame, label

    def process_frames_single(self, image_dir):
        image_paths = sorted(os.listdir(image_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        keypoints_array = torch.zeros((len(image_paths), 33, 2), dtype=torch.float32)

        for i, img_path in enumerate(image_paths):
            img = cv2.imread(os.path.join(image_dir, img_path))
            img = resize_image(img, (1920, 1080))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            keypoints = get_closest_person_keypoints(img)
            # keypoints = torch.from_numpy(keypoints)
            keypoints_array[i, :, :] = keypoints

        # keypoints_array = torch.tensor(keypoints_array, dtype=torch.float32)
        return keypoints_array


# CUDA 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureProcessor(nn.Module):
    def __init__(self, embed_dim=120):
        super(FeatureProcessor, self).__init__()
        self.embed_dim = embed_dim
        self.linear_proj = nn.Linear(30*4, embed_dim).to("cuda")  # Frame별 Feature Embedding

    def forward(self, x):
        batch, frames, _ = x.shape  # (Batch, Frames, 2, 15)
        x = x.unsqueeze(-1).repeat(1, 1, 1, 4).to("cuda")
        x = x.view(batch, frames, -1).to("cuda")  # (Batch, Frames, 120)
        x = self.linear_proj(x)  # (Batch, Frames, embed_dim)

        return x.to("cuda")

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=120, num_heads=4):
        super(CrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads).to("cuda")

    def forward(self, instructor, child):
        max_seq_len = max(instructor.shape[1], child.shape[1])

        # Zero padding을 적용하여 길이 맞추기
        pad_instructor = torch.zeros((instructor.shape[0], max_seq_len, instructor.shape[2]), device="cuda")
        pad_child = torch.zeros((child.shape[0], max_seq_len, child.shape[2]), device="cuda")

        pad_instructor[:, :instructor.shape[1], :] = instructor.to("cuda")
        pad_child[:, :child.shape[1], :] = child.to("cuda")

        # Multi-Head Attention 적용
        attn_output, _ = self.cross_attn(pad_instructor, pad_child, pad_child)

        return attn_output.to("cuda")

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=120, num_heads=4):
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads).to("cuda")
        # self.qkv = nn.Linear(embed_dim, embed_dim * 3)

    def forward(self, x):
        # qkv = self.qkv(x)
        # print(f"The size of the feature map: {qkv.shape}")
        # query, key, value = qkv[]
        attn_output, _ = self.self_attn(x.to("cuda"), x.to("cuda"), x.to("cuda"))
        return attn_output.to("cuda")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=120, num_classes=1, ch=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, (out.size(2), out.size(3)))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = self.sigmoid(out)
        return out

class Classifier(nn.Module):
    def __init__(self, embed_dim=120, num_classes=1):
        super(Classifier, self).__init__()
        self.classifier = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=embed_dim, num_classes=1).to("cuda")
        # self.fc = nn.Linear(embed_dim, num_classes).to("cuda")

    def forward(self, x):
        # x = x.mean(dim=1).to("cuda")  # Frame-wise 평균
        # x = self.fc(x)
        x = self.classifier(x)
        return x.to("cuda")

class Extractor(nn.Module):
    def __init__(self, pose):
        super(Extractor, self).__init__()
        self.device = "cuda"
        self.pose = pose

    def forward(self, child_batch, teacher_batch):
        map = torch.zeros((child_batch.shape[0], child_batch.shape[1], teacher_batch.shape[1]), dtype=torch.float32)
        child_features = self.vectorizing(self.filtering(self.matrix(child_batch)))
        teacher_features = self.vectorizing(self.filtering(self.matrix(teacher_batch)))
        for it in range(child_features.shape[0] if child_features.shape[0] == teacher_features.shape[0] else exit()):
            map[it, :, :] = torch.matmul(child_features[it, :, :], teacher_features[it, :, :].T)
            # map[it, :, :] = torch.nn.functional.normalize(map[it, :, :], p=2, dim=-1)
        # return child_features.to("cuda"), teacher_features.to("cuda")
        return map

    def matrix(self, keypoints):
        keypoints = keypoints.transpose(-2, -1)
        return keypoints.to("cuda")

    def filtering(self, keypoints):
        M_s = torch.zeros((33, 6), dtype=torch.float32, device="cuda")
        M_s[0, 0] = 1
        M_s[11, 1] = M_s[12, 1] = 0.5
        M_s[13, 2] = M_s[14, 3] = 1
        M_s[15, 4] = M_s[16, 5] = 1

        K_sk = torch.matmul(keypoints.to("cuda"), M_s.to("cuda"))
        # K_sk = torch.nn.functional.normalize(K_sk, p=2, dim=-2)
        return K_sk.to("cuda")

    def vectorizing(self, K_sk):
        M_d = torch.zeros((6, 15), dtype=torch.float32, device="cuda")
        M_d[0, :5] = -1
        M_d[1, 5:9] = -1
        M_d[2, 9:12] = -1
        M_d[3, 12:14] = -1
        M_d[4, 14] = -1
        M_d[1, 0] = M_d[2, 1] = M_d[3, 2] = M_d[4, 3] = M_d[5, 4] = 1
        M_d[2, 5] = M_d[3, 6] = M_d[4, 7] = M_d[5, 8] = 1
        M_d[3, 9] = M_d[4, 10] = M_d[5, 11] = 1
        M_d[4, 12] = M_d[5, 13] = 1
        M_d[5, 14] = 1

        K_final = torch.matmul(K_sk.to("cuda"), M_d.to("cuda"))
        # K_final = torch.nn.functional.normalize(K_final, p=2, dim=-1)
        K_final = K_final.flatten(-2, -1)
        K_final = torch.nn.functional.normalize(K_final, p=2, dim=-1)
        return K_final.to("cuda")

class AttentionImitationNetwork(nn.Module):
    def __init__(self, embed_dim=120, num_heads=4, pose=pose, num_classes=1):
        super(AttentionImitationNetwork, self).__init__()
        self.extractor = Extractor().to("cuda")
        # self.feature_processor = FeatureProcessor(embed_dim).to("cuda")
        # self.cross_attention = CrossAttention(embed_dim, num_heads).to("cuda")
        # self.self_attention = SelfAttention(embed_dim, num_heads).to("cuda")
        # self.classifier = Classifier(embed_dim, num_classes).to("cuda")
        self.classifier = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_classes).to("cuda")

    def forward(self, child, instructor):
        # child_feat, instructor_feat = self.extractor(child.to("cuda"), instructor.to("cuda"))
        map = self.extractor(child.to("cuda"), instructor.to("cuda"))
        # print(f"Feature dimension of child_feat before feature processor: {child_feat.shape}")
        # print(f"Feature dimension of instructor_feat before feature processor: {instructor_feat.shape}")
        # instructor_feat = self.feature_processor(instructor_feat.to("cuda")).to("cuda")
        # child_feat = self.feature_processor(child_feat.to("cuda")).to("cuda")
        # print(f"Feature dimension of child_feat after feature processor: {child_feat.shape}")
        # print(f"Feature dimension of instructor_feat after feature processor: {instructor_feat.shape}")

        # cross_attn_out = self.cross_attention(instructor_feat, child_feat).to("cuda")
        # print(f"Feature dimension after cross attention: {cross_attn_out.shape}")
        # self_attn_out = self.self_attention(cross_attn_out).to("cuda")
        # print(f"Feature dimension after self attention: {self_attn_out.shape}")

        # self_attn_out = self_attn_out.unsqueeze(0)
        # print(self_attn_out.shape)
        # self_attn_out = self_attn_out.unsqueeze(-1).permute(0, 2, 3, 1)

        # print(self_attn_out.shape)
        # out = self.classifier(self_attn_out)
        map = similarity_map_to_rgb_tensor_no_plt(map)
        # map = apply_colormap(map)
        # print(map.shape)
        out = self.classifier(map)
        # print(out.shape)
        return out.to("cuda")
