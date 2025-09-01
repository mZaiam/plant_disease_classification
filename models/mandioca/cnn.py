import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import transforms
import torchvision.models as models
from optuna import TrialPruned

class CNN(nn.Module):
    '''
    Convolutional Neural Network (CNN).

    Args:
        input_conv_filters: int for number of channels of inputs.
        conv_filters:       list with number of filters for each convolutional layer.
        conv_activation:    torch activation function.
        conv_kernel_size:   int with kernel size.
        conv_batchnorm:     bool for BatchNorm2d inclusion.
        pool_kernel_size:   int with pooling kernel size.
        adaptive_pool_size: int with adaptive pooling size.
        lin_neurons:        list with number of neurons for each linear layer.
        lin_activation:     torch activation function.    
        lin_batchnorm:      bool for BatchNorm1d inclusion.
        device:             torch device.
    '''
    
    def __init__(
        self,
        input_conv_filters=3,
        conv_filters=[16, 32, 64, 128, 256],
        conv_activation=nn.ReLU(),
        conv_kernel_size=7,
        conv_batchnorm=True,
        pool_kernel_size=2,
        adaptive_pool_size=4,
        lin_neurons=[128, 32, 4],
        lin_activation=nn.ReLU(),     
        lin_batchnorm=True,
        device='cpu',
    ):
        super(CNN, self).__init__()

        self.device = device
        
        # Convolutional layers
        conv_layers = []
        
        for i in range(len(conv_filters)):
            input_filters = input_conv_filters if i == 0 else conv_filters[i - 1] 
            
            conv_layers.append(
                nn.Conv2d(
                    in_channels=input_filters, 
                    out_channels=conv_filters[i],
                    kernel_size=conv_kernel_size,
                ),
            )
            
            if conv_batchnorm:
                conv_layers.append(nn.BatchNorm2d(conv_filters[i]))
                
            conv_layers.append(conv_activation)
            
            if i != len(conv_filters) - 1:
                conv_layers.append(nn.AvgPool2d(pool_kernel_size))
            
        self.conv = nn.Sequential(*conv_layers)
        
        # Adaptive Pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            conv_out = self.conv(x)
            pool_out = self.flatten(self.adaptive_pool(conv_out))
            input_lin_neurons = pool_out.shape[-1]
        
        # Linear layers
        lin_layers = []
        
        for i in range(len(lin_neurons)):
            input_neurons = input_lin_neurons if i == 0 else lin_neurons[i - 1]
            
            lin_layers.append(
                nn.Linear(
                    in_features=input_neurons, 
                    out_features=lin_neurons[i],
                ),
            )
            
            if lin_batchnorm:
                if i != len(lin_neurons) - 1:
                    lin_layers.append(nn.BatchNorm1d(lin_neurons[i]))
                
            if i != len(lin_neurons) - 1:
                lin_layers.append(lin_activation)
        
        self.lin = nn.Sequential(*lin_layers)
        
    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.flatten(self.adaptive_pool(conv_out))
        lin_out = self.lin(pool_out)
        return lin_out

    def fit(
        self,
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        epochs, 
        trial=None,
        verbose=False, 
        optuna=False
    ):
        '''
        Training method.

        Args:
            train_loader: torch DataLoader with train data. 
            val_loader:   torch DataLoader with test data.
            optimizer:    torch optimizer.
            criterion:    torch criterion.
            epochs:       int with number of epochs.
            trial:        optuna trial for pruning in HP tunning.
            verbose:      bool for loss value outputs in training.
            optuna:       bool for optuna usage.
        '''
        
        self.to(self.device)
        
        loss_train = []
        loss_val = []
        acc_train = []
        acc_val = []
        best_loss = 0.0

        for i, epoch in enumerate(range(epochs)):
            # Train data
            self.train()  
            loss_train_epoch = 0.0
            total_correct_train = 0
            total_samples_train = 0
    
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()  
                y_pred = self.forward(x_batch)  
                loss = criterion(y_pred, y_batch)  
                loss.backward() 
                optimizer.step() 
                loss_train_epoch += loss.item()
                
                preds = y_pred.argmax(dim=1)
                total_correct_train += (preds == y_batch).sum().item()
                total_samples_train += y_batch.size(0)
    
            loss_train_epoch /= len(train_loader)
            loss_train.append(loss_train_epoch)
            acc_train_epoch = 100 * total_correct_train / total_samples_train
            acc_train.append(acc_train_epoch)
    
            # Val data
            self.eval()  
            loss_val_epoch = 0.0
            total_correct_val = 0
            total_samples_val = 0
            
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.forward(x_batch) 
                    loss = criterion(y_pred, y_batch) 
                    loss_val_epoch += loss.item()
                    
                    preds = y_pred.argmax(dim=1)
                    total_correct_val += (preds == y_batch).sum().item()
                    total_samples_val += y_batch.size(0)
    
            loss_val_epoch /= len(val_loader)
            loss_val.append(loss_val_epoch)
            acc_val_epoch = 100 * total_correct_val / total_samples_val 
            acc_val.append(acc_val_epoch)

            # Pruning 
            if trial:
                trial.report(acc_val_epoch, step=i)
                if trial.should_prune():
                    raise TrialPruned()

            # Loss printing
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}: train_acc: {acc_train_epoch:.2f} val_acc: {acc_val_epoch:.2f}')
                
        self.loss_train = loss_train
        self.loss_val = loss_val
        self.acc_train = acc_train
        self.acc_val = acc_val
        self.model_state = self.state_dict()

        # Return losses for optuna HP tuning
        if optuna:
            return self.loss_val[-1], (self.acc_train, self.acc_val)

class DatasetAugmentation(torch.utils.data.Dataset):
    '''
    Dataset Augmentation.

    Args:
        tensors:   tuple with x and y data.
        transform: torchvision.transforms with data augmentation transforms.
    '''
    def __init__(self, tensors, transform=None):
        self.x, self.y = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.x)

class SubsetAugmentation(torch.utils.data.Dataset):
    '''
    Dataset Augmentation for Subsets.

    Args:
        dataset:   torch dataset.
        idxs:      indices for subset.
        transform: torchvision.transforms with data augmentation transforms.
    '''
    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = idxs
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[self.idxs[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.idxs)

# ResNet classes

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNet(nn.Module):
    def __init__(self, n_classes, fine_tuning=False, device='cpu'):
        super(ResNet, self).__init__() 

        self.device = device
        self.fine_tuning = fine_tuning
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = Identity()
        self.resnet = resnet.to(device)
        
        self.fc = nn.Linear(
            in_features=2048,
            out_features=n_classes
        ).to(device)

    def forward(self, x):
        if self.fine_tuning:
            resnet_out = self.resnet(x)
        else:
            with torch.no_grad():
                resnet_out = self.resnet(x)
        lin_out = self.fc(resnet_out)
        return lin_out

    def fit(
        self,
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        epochs, 
        trial=None,
        verbose=False, 
        optuna=False
    ):
        '''
        Training method.

        Args:
            train_loader: torch DataLoader with train data. 
            val_loader:   torch DataLoader with test data.
            optimizer:    torch optimizer.
            criterion:    torch criterion.
            epochs:       int with number of epochs.
            trial:        optuna trial for pruning in HP tunning.
            verbose:      bool for loss value outputs in training.
            optuna:       bool for optuna usage.
        '''
        
        self.to(self.device)
        
        loss_train = []
        loss_val = []
        acc_train = []
        acc_val = []
        best_loss = 0.0

        for i, epoch in enumerate(range(epochs)):
            # Train data
            self.train()  
            loss_train_epoch = 0.0
            total_correct_train = 0
            total_samples_train = 0
    
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()  
                y_pred = self.forward(x_batch)  
                loss = criterion(y_pred, y_batch)  
                loss.backward() 
                optimizer.step() 
                loss_train_epoch += loss.item()
                
                preds = y_pred.argmax(dim=1)
                total_correct_train += (preds == y_batch).sum().item()
                total_samples_train += y_batch.size(0)
    
            loss_train_epoch /= len(train_loader)
            loss_train.append(loss_train_epoch)
            acc_train_epoch = 100 * total_correct_train / total_samples_train
            acc_train.append(acc_train_epoch)
    
            # Val data
            self.eval()  
            loss_val_epoch = 0.0
            total_correct_val = 0
            total_samples_val = 0
            
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.forward(x_batch) 
                    loss = criterion(y_pred, y_batch) 
                    loss_val_epoch += loss.item()
                    
                    preds = y_pred.argmax(dim=1)
                    total_correct_val += (preds == y_batch).sum().item()
                    total_samples_val += y_batch.size(0)
    
            loss_val_epoch /= len(val_loader)
            loss_val.append(loss_val_epoch)
            acc_val_epoch = 100 * total_correct_val / total_samples_val 
            acc_val.append(acc_val_epoch)
            
            # Pruning 
            if trial:
                trial.report(acc_val_epoch, step=i)
                if trial.should_prune():
                    raise TrialPruned()

            # Loss printing
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}: train_acc: {acc_train_epoch:.2f} val_acc: {acc_val_epoch:.2f}')
                
        self.loss_train = loss_train
        self.loss_val = loss_val
        self.acc_train = acc_train
        self.acc_val = acc_val
        self.model_state = self.state_dict()

        # Return losses for optuna HP tuning
        if optuna:
            return self.loss_val[-1], (self.acc_train, self.acc_val)

# Transfer Learning class

def instantiate(trial, device):
    '''
    Model instantiation.
    
    Args:
        trial:        optuna trial.
        device:       torch device.
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
    N_CLASSES = 38
        
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

    return cnn

class CNNTransferLearning(nn.Module):
    def __init__(self, best_trial, n_classes, fine_tuning=False, device='cpu', path='../plantvillage/best_models/plantvillage.pth'):
        super(CNNTransferLearning, self).__init__() 

        self.device = device
        self.fine_tuning = fine_tuning
        
        cnn = instantiate(
            best_trial,
            device,
        )
        cnn.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model'])
        in_features = cnn.lin[-1].weight.shape[-1]
        cnn.lin[-1] = Identity()
        self.cnn = cnn.to(device)
        
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=n_classes
        ).to(device)

    def forward(self, x):
        if self.fine_tuning:
            cnn_out = self.cnn(x)
        else:
            with torch.no_grad():
                cnn_out = self.cnn(x)
        lin_out = self.fc(cnn_out)
        return lin_out

    def fit(
        self,
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        epochs, 
        trial=None,
        verbose=False, 
        optuna=False
    ):
        '''
        Training method.

        Args:
            train_loader: torch DataLoader with train data. 
            val_loader:   torch DataLoader with test data.
            optimizer:    torch optimizer.
            criterion:    torch criterion.
            epochs:       int with number of epochs.
            trial:        optuna trial for pruning in HP tunning.
            verbose:      bool for loss value outputs in training.
            optuna:       bool for optuna usage.
        '''
        
        self.to(self.device)
        
        loss_train = []
        loss_val = []
        acc_train = []
        acc_val = []
        best_loss = 0.0

        for i, epoch in enumerate(range(epochs)):
            # Train data
            self.train()  
            loss_train_epoch = 0.0
            total_correct_train = 0
            total_samples_train = 0
    
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()  
                y_pred = self.forward(x_batch)  
                loss = criterion(y_pred, y_batch)  
                loss.backward() 
                optimizer.step() 
                loss_train_epoch += loss.item()
                
                preds = y_pred.argmax(dim=1)
                total_correct_train += (preds == y_batch).sum().item()
                total_samples_train += y_batch.size(0)
    
            loss_train_epoch /= len(train_loader)
            loss_train.append(loss_train_epoch)
            acc_train_epoch = 100 * total_correct_train / total_samples_train
            acc_train.append(acc_train_epoch)
    
            # Val data
            self.eval()  
            loss_val_epoch = 0.0
            total_correct_val = 0
            total_samples_val = 0
            
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.forward(x_batch) 
                    loss = criterion(y_pred, y_batch) 
                    loss_val_epoch += loss.item()
                    
                    preds = y_pred.argmax(dim=1)
                    total_correct_val += (preds == y_batch).sum().item()
                    total_samples_val += y_batch.size(0)
    
            loss_val_epoch /= len(val_loader)
            loss_val.append(loss_val_epoch)
            acc_val_epoch = 100 * total_correct_val / total_samples_val 
            acc_val.append(acc_val_epoch)
            
            # Pruning 
            if trial:
                trial.report(acc_val_epoch, step=i)
                if trial.should_prune():
                    raise TrialPruned()

            # Loss printing
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}: train_acc: {acc_train_epoch:.2f} val_acc: {acc_val_epoch:.2f}')
                
        self.loss_train = loss_train
        self.loss_val = loss_val
        self.acc_train = acc_train
        self.acc_val = acc_val
        self.model_state = self.state_dict()

        # Return losses for optuna HP tuning
        if optuna:
            return self.loss_val[-1], (self.acc_train, self.acc_val)
