import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import pandas as pd
import optuna
import numpy as np
import os
from PIL import Image

from optuna_funcs import load_data, cross_validation_resnet
from cnn import ResNet, DatasetAugmentation, SubsetAugmentation

# Seed and device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
print(f'Device: {device}')

# Load data

images, labels, names = load_data(name='milho', root="../../digipathos/")

# Train-test-validation split

TEST_SIZE = 0.1
VAL_SIZE = 0.1

x, y = images.numpy(), labels.numpy()
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=VAL_SIZE, stratify=y_train_val, random_state=SEED)

x_train_val = torch.tensor(x_train_val, dtype=torch.float32)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_train_val = torch.tensor(y_train_val, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Creating Datasets and DataLoaders with WeightedRandomSampler

class_counts = torch.bincount(y_train)
weights = 1.0 / class_counts.float()
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

train_val_dataset = TensorDataset(x_train_val, y_train_val)

train_dataset = DatasetAugmentation((x_train, y_train), transform=train_transforms)
val_dataset = DatasetAugmentation((x_val, y_val), transform=val_transforms)
test_dataset = DatasetAugmentation((x_test, y_test), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Cross-Validation

STUDY_NAME = "milho_resnet_fine_tuning"
n_classes = 5
fine_tuning = True

kfold_acc = cross_validation_resnet(
    train_val_dataset,
    train_transforms, 
    val_transforms,
    device,
    n_classes=n_classes,
    fine_tuning=fine_tuning
)

# Saving best model

train_val_dataset = DatasetAugmentation((x_train_val, y_train_val), transform=train_transforms)
train_val_loader = DataLoader(train_val_dataset, batch_size=8, sampler=sampler)

resnet = ResNet(n_classes=n_classes, fine_tuning=fine_tuning)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=5e-4)

_, (acc_train, acc_test) = resnet.fit(
    train_val_loader, 
    test_loader, 
    optimizer,
    criterion,
    epochs=30,
    optuna=True
)

train_loss, test_loss = resnet.loss_train, resnet.loss_val
train_acc, test_acc = resnet.acc_train, resnet.acc_val

model_state = resnet.model_state

best_model = {
    'model': model_state,
    'train_loss': train_loss,
    'test_loss': test_loss,
    'train_acc': train_acc,
    'test_acc': test_acc,
    'kfold_acc': kfold_acc
}
torch.save(best_model, f'best_models/{STUDY_NAME}.pth')
