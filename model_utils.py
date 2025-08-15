# model_utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
from timm.data import create_transform
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from flwr.common import ndarrays_to_parameters
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)
    return model

def get_loaders(client_id=None, num_clients=2):
    transform_train = create_transform(
    input_size=32,
    is_training=True,
    mean=(0.5071, 0.4865, 0.4409),
    std=(0.2673, 0.2564, 0.2762),
    auto_augment='rand-m9-mstd0.5-inc1',  # stronger augmentation
    re_prob=0.25, re_mode='pixel', re_count=1
)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    if client_id is not None:  
        # --- Create per-class index buckets ---
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(train_data):
            class_indices[label].append(idx)

        # --- Select subset for this client ---
        client_indices = []
        for cls in range(10):
            indices = class_indices[cls]
            indices.sort()
            shard_size = len(indices) // num_clients
            start = client_id * shard_size
            end = start + shard_size if client_id != num_clients - 1 else len(indices)
            client_indices.extend(indices[start:end])

        train_subset = Subset(train_data, client_indices)
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1000)

        print(f"[Client {client_id}] Got {len(train_subset)} samples from all classes")
    else:
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1000)
    return train_loader, test_loader

def train_one_epoch(model, dataloader, device=None):
    model.train()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 💥 move model to device

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()

def evaluate_model(model, dataloader, device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct, total, loss_total = 0, 0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)  # 💥 move data to same device
            out = model(x)
            loss = criterion(out, y)
            batch_size = x.size(0)
            loss_total += loss.item() * batch_size  # multiply by batch size
            _, predicted = torch.max(out.data, 1)
            total += batch_size
            correct += (predicted == y).sum().item()

    accuracy = correct / total * 100
    avg_loss = loss_total / total  # average loss per sample
    return accuracy, avg_loss

def get_initial_parameters():
    model = get_model()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(weights)
