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

from optuna_funcs import load_data, cross_validation_tl
from cnn import CNNTransferLearning, DatasetAugmentation, SubsetAugmentation

# Seed and device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
print(f'Device: {device}')

# Load data

images, labels, names = load_data(name='feijao', root="../../digipathos/")

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

train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Loading best trial

STUDY_NAME = 'plantvillage'
study = optuna.load_study(study_name=STUDY_NAME, storage=f'sqlite:///../plantvillage/hp_tuning/{STUDY_NAME}.db')

# Reading Cross-Validation

df = pd.read_csv(f'../plantvillage/cross_validation/{STUDY_NAME}.csv')
columns = df.filter(like="Cross_Val")
df['Mean'] = columns.mean(axis=1)
df = df.drop(columns="Unnamed: 0")
df = df.sort_values(by=["Mean"], ascending=False)
df = df.reset_index(drop=True)
print(f'Best accuracy: {float(df["Mean"][0]):.2f}%')

# Best model

best_trial_num = df["Number"][0]
best_trial = study.get_trials()[best_trial_num]

# Cross-Validation

STUDY_NAME = "feijao_plantvillage_domain_adaptation"
n_classes = 4
fine_tuning = False

kfold_acc = cross_validation_tl(
    best_trial,
    train_val_dataset,
    train_transforms, 
    val_transforms,
    device,
    n_classes=n_classes,
    fine_tuning=fine_tuning
)

# Saving best model

train_val_dataset = DatasetAugmentation((x_train_val, y_train_val), transform=train_transforms)
train_val_loader = DataLoader(train_val_dataset, batch_size=4, sampler=sampler)

cnn = CNNTransferLearning(best_trial, n_classes=n_classes, fine_tuning=fine_tuning, device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=5e-4)

_, (acc_train, acc_test) = cnn.fit(
    train_val_loader, 
    test_loader, 
    optimizer,
    criterion,
    epochs=30,
    optuna=True
)

train_loss, test_loss = cnn.loss_train, cnn.loss_val
train_acc, test_acc = cnn.acc_train, cnn.acc_val

model_state = cnn.model_state

best_model = {
    'model': model_state,
    'train_loss': train_loss,
    'test_loss': test_loss,
    'train_acc': train_acc,
    'test_acc': test_acc,
    'kfold_acc': kfold_acc
}
torch.save(best_model, f'best_models/{STUDY_NAME}.pth')
