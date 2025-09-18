import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from sklearn.model_selection import KFold
import pandas as pd
import optuna
import numpy as np
import os
from PIL import Image

from cnn import CNN, CNNTransferLearning, ResNet, DatasetAugmentation, SubsetAugmentation

def load_data(name, root, img_size=(224, 224)):
    '''
    Load data from specified root directory.

    Args:
        name:     str for directory matching.
        root:     str with root directory.
        img_size: tuple for image resize.
    
    Returns:
        images: torch.tensor with images.
        labels: torch.tensor with labels.
        names:  list with strs for class names.
    '''
    data = {}
    i = 0
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  
    ])
    
    directories = os.listdir(root)
    
    for d in directories:
        images = []
        if name in d:
            class_path = os.path.join(root, d)
            for im in os.listdir(class_path):
                im_path = os.path.join(class_path, im)
                image = transform(Image.open(im_path).convert('RGB'))
                images.append(image)
            data[d] = torch.stack(images, dim=0)

    images = []
    labels = []
    names = []
    for i, c in enumerate(list(data.keys())):
        images.append(data[c])
        labels.append(i * torch.ones(data[c].shape[0]))
        names.extend(data[c].shape[0] * [c])
    
    images = torch.concatenate(images, dim=0)
    labels = torch.concatenate(labels, dim=0)

    return images, labels, names

def objective(trial, train_loader, val_loader, device, kfold=False, best_model=False, instantiate=True):
    '''
    Optuna objective.
    
    Args:
        trial:        optuna trial.
        train_loader: torch DataLoader with train data.
        val_loader:   torch DataLoader with val data.
        device:       torch device.
        kfold:        bool for KFold usage.
        best_model:   bool for returning best model.
        instantiate:  bool for just instantiating the model.
    '''

    # Defining HP grid
    params = {
        'conv_layers': trial.suggest_int('conv_layers', 1, 5),
        'conv_activation': trial.suggest_categorical('conv_activation', ['relu', 'sigmoid', 'tanh', 'leaky_relu']),
        'conv_kernel_size': trial.suggest_categorical('conv_kernel_size', [3, 5, 7]),
        'conv_batchnorm': trial.suggest_categorical('conv_batchnorm', [True, False]),
        'adaptive_pool_size': trial.suggest_categorical('adaptive_pool_size', [1, 2, 3, 4]),
        'lin_layers': trial.suggest_int('lin_layers', 1, 3),
        'lin_activation': trial.suggest_categorical('lin_activation', ['relu', 'sigmoid', 'tanh']),
        'lin_batchnorm': trial.suggest_categorical('lin_batchnorm', [True, False]),
        'lr': trial.suggest_categorical('lr', [5e-2, 1e-3, 5e-3, 1e-4]),
    }
    
    # Instantianting CNN
    N_CLASSES = 5
        
    dict_activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU()
    }
        
    list_conv_layers = [
        [16],
        [16, 32],
        [16, 32, 64],
        [16, 32, 64, 128],
        [16, 32, 64, 128, 256],
    ]
    
    list_lin_layers = [
        [128, N_CLASSES],
        [128, 64, N_CLASSES],
        [128, 64, 32, N_CLASSES],
    ]
        
    cnn = CNN(
        conv_filters=list_conv_layers[params['conv_layers'] - 1],
        conv_activation=dict_activations[params['conv_activation']],
        conv_kernel_size=params['conv_kernel_size'],
        conv_batchnorm=params['conv_batchnorm'],
        adaptive_pool_size=params['adaptive_pool_size'],
        lin_neurons=list_lin_layers[params['lin_layers'] - 1],
        lin_activation=dict_activations[params['lin_activation']],     
        lin_batchnorm=params['lin_batchnorm'],
        device=device
    )

    if instantiate:
        return cnn

    else:
        # Instantiating optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=params['lr'])
        
        # Training model
        EPOCHS = 30
        
        loss, (train_acc, val_acc) = cnn.fit(
            train_loader,
            val_loader,
            optimizer,
            criterion,
            EPOCHS,
            trial=trial,
            optuna=True
        )
    
        print(f'train_acc: {max(train_acc):.2f} val_acc: {max(val_acc):.2f}')
    
        if best_model:
            return cnn
        else:
            if kfold:
                return val_acc[-1]
            else:
                return loss

def cross_validation(study, dataset, train_transforms, val_transforms, device, n_splits=10, top=50):
    '''
    Cross-validation.

    Args:
        study:            optuna trial.
        dataset:          torch TensorDataset with train+val data.
        train_transforms: torchvision transforms for train data.
        val_transforms:   torchvision transforms for val data.
        device:           torch device.
        n_splits:         int for KFold splits.
        top:              int for number of top models to be analyzed. 
    
    Returns:
        params:           parameters for best trial.
        top_50_loss:      loss lists for top 50 models.
        top_50_number:    trial number for top 50 models.
    
    Ref:
        https://saturncloud.io/blog/how-to-use-kfold-cross-validation-with-dataloaders-in-pytorch/
    '''
    
    trials = [i for i in study.trials if i.values is not None]
    best_trials = sorted(trials, key=lambda t: t.values)[:top]

    top_50_loss = {}
    top_50_mean = {}
    top_50_params = {}
    top_50_number = {}

    # KFold Cross-Validation
    for num, trial in enumerate(best_trials):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f'Fold {fold+1}')

            train_subset = SubsetAugmentation(dataset, train_idx, train_transforms)
            val_subset = SubsetAugmentation(dataset, val_idx, val_transforms)
            
            train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
            
            acc = objective(trial, train_loader, val_loader, device, kfold=True, instantiate=False)
            losses.append(acc)

        mean = sum(losses) / len(losses)

        top_50_params.update({f"Modelo {num}": trial})
        top_50_loss.update({f"Modelo {num}": losses})
        top_50_mean.update({f"Modelo {num}": mean})
        top_50_number.update({f"Modelo {num}": {"Number": trial.number}})

    best_model = min(top_50_mean, key=top_50_mean.get)
    params = top_50_params[best_model]

    return params, top_50_loss, top_50_number

def cross_validation_resnet(dataset, train_transforms, val_transforms, device, n_splits=10, epochs=30, n_classes=5, fine_tuning=False):
    '''
    Cross-validation.

    Args:
        dataset:          torch TensorDataset with train+val data.
        train_transforms: torchvision transforms for train data.
        val_transforms:   torchvision transforms for val data.
        device:           torch device.
        n_splits:         int for KFold splits.
        epochs:           int for epochs in training.
        n_classes:        int for classes in dataset.
        fine_tuning:      bool for fine tuning.
    
    Returns:
        acc:              list with accuracies from each Fold.      
    
    Ref:
        https://saturncloud.io/blog/how-to-use-kfold-cross-validation-with-dataloaders-in-pytorch/
    '''

    # KFold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold+1}')

        resnet = ResNet(n_classes, device=device, fine_tuning=fine_tuning)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(resnet.parameters(), lr=5e-4)

        train_subset = SubsetAugmentation(dataset, train_idx, train_transforms)
        val_subset = SubsetAugmentation(dataset, val_idx, val_transforms)
        
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
        
        _, (acc_train, acc_val) = resnet.fit(
            train_loader, 
            val_loader, 
            optimizer,
            criterion,
            epochs,
            optuna=True
        )
        acc.append(acc_val[-1])
        print(acc_val)

    return acc

def cross_validation_tl(best_trial, dataset, train_transforms, val_transforms, device, n_splits=10, epochs=30, n_classes=8, fine_tuning=False):
    '''
    Cross-validation.

    Args:
        best_trial:       optuna trial with model params.
        dataset:          torch TensorDataset with train+val data.
        train_transforms: torchvision transforms for train data.
        val_transforms:   torchvision transforms for val data.
        device:           torch device.
        n_splits:         int for KFold splits.
        epochs:           int for epochs in training.
        n_classes:        int for classes in dataset.
        fine_tuning:      bool for fine tuning.
    
    Returns:
        acc:              list with accuracies from each Fold.      
    
    Ref:
        https://saturncloud.io/blog/how-to-use-kfold-cross-validation-with-dataloaders-in-pytorch/
    '''

    # KFold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold+1}')

        cnn = CNNTransferLearning(best_trial, n_classes, device=device, fine_tuning=fine_tuning)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=5e-4)

        train_subset = SubsetAugmentation(dataset, train_idx, train_transforms)
        val_subset = SubsetAugmentation(dataset, val_idx, val_transforms)
        
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
        
        _, (acc_train, acc_val) = cnn.fit(
            train_loader, 
            val_loader, 
            optimizer,
            criterion,
            epochs,
            optuna=True
        )
        acc.append(acc_val[-1])
        print(acc_val)

    return acc
