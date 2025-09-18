import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split

import pandas as pd
import optuna
import numpy as np
import os
from PIL import Image

from cnn import CNN, DatasetAugmentation
from optuna_funcs import load_data, objective

# Seed and device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
print(f'Device: {device}')

# Load data

images, labels, names = load_data(name='milho', root="../../digipathos/")

# Train-test-validation split

TEST_SIZE = 0.1

x, y = images.numpy(), labels.numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Creating Datasets and DataLoaders with WeightedRandomSampler

class_counts = torch.bincount(y_train)
weights = 1.0 / class_counts.float()
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

train_dataset = DatasetAugmentation((x_train, y_train), transform=train_transforms)
test_dataset = DatasetAugmentation((x_test, y_test), transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Loading Optuna

STUDY_NAME = 'milho_baseline'
study = optuna.load_study(study_name=STUDY_NAME, storage=f'sqlite:///hp_tuning/{STUDY_NAME}.db')

# Reading Cross-Validation

df = pd.read_csv(f'cross_validation/{STUDY_NAME}.csv')
columns = df.filter(like="Cross_Val")
df['Mean'] = columns.mean(axis=1)
df = df.drop(columns="Unnamed: 0")
df = df.sort_values(by=["Mean"], ascending=False)
df = df.reset_index(drop=True)
print(f'Best accuracy: {float(df["Mean"][0]):.2f}%')

# Best model

best_trial_num = df["Number"][0]
best_trial = study.get_trials()[best_trial_num]
kfold_acc = [df[f'Cross_Val_{i}'][0].item() for i in range(10)]

# Training best model

cnn = objective(
    best_trial,
    train_loader,
    test_loader,
    device,
    best_model=True,
    instantiate=False
)

train_loss, test_loss = cnn.loss_train, cnn.loss_val
train_acc, test_acc = cnn.acc_train, cnn.acc_val

model_state = cnn.model_state

# Save model

best_model = {
    'model': model_state,
    'train_loss': train_loss,
    'test_loss': test_loss,
    'train_acc': train_acc,
    'test_acc': test_acc,
    'kfold_acc': kfold_acc
}
torch.save(best_model, f'best_models/{STUDY_NAME}.pth')
