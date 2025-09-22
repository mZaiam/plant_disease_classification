import captum
import torch
import optuna
import pandas as pd
import numpy as np

from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from skimage.segmentation import slic

from optuna_funcs import load_data, objective
from cnn import CNN, ResNet, CNNTransferLearning

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
print(f'Device: {device}')

baseline = torch.load('../best_models/milho_baseline.pth', map_location='cpu', weights_only=True)
plantvillage_da = torch.load('../best_models/milho_plantvillage_domain_adaptation.pth', map_location='cpu', weights_only=True)
plantvillage_ft = torch.load('../best_models/milho_plantvillage_fine_tuning.pth', map_location='cpu', weights_only=True)
resnet_da = torch.load('../best_models/milho_resnet_domain_adaptation.pth', map_location='cpu', weights_only=True)
resnet_ft = torch.load('../best_models/milho_resnet_fine_tuning.pth', map_location='cpu', weights_only=True)

images, labels, names = load_data(name='milho', root="../../../digipathos/")

TEST_SIZE = 0.1

x, y = images.numpy(), labels.numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
names_train, names_test = train_test_split(names, test_size=TEST_SIZE, stratify=names, random_state=SEED)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

n_classes = 5
results = {}

models_to_process = {
    'baseline': baseline,
    'plantvillage_da': plantvillage_da,
    'plantvillage_ft': plantvillage_ft,
    'resnet_da': resnet_da,
    'resnet_ft': resnet_ft,
}

def deprocess_image(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

for name, state_dict in models_to_process.items():
    print(f"Model: {name}")

    if 'resnet' in name:
        model = ResNet(n_classes=5, fine_tuning=True, device=device).to(device)
    else:
        STUDY_NAME = 'milho_baseline' if name == 'baseline' else 'plantvillage'
        study = optuna.load_study(study_name=STUDY_NAME, storage=f'sqlite:///{"../hp_tuning" if name == "baseline" else "../../plantvillage/hp_tuning"}/{STUDY_NAME}.db')
        df = pd.read_csv(f'{"../cross_validation" if name == "baseline" else "../../plantvillage/cross_validation"}/{STUDY_NAME}.csv')
        df['Mean'] = df.filter(like="Cross_Val").mean(axis=1)
        best_trial_num = df.sort_values(by=["Mean"], ascending=False).reset_index(drop=True).loc[0, "Number"]
        best_trial = study.get_trials()[best_trial_num]
        if name == 'baseline':
            model = objective(best_trial, None, None, device=device, instantiate=True).to(device)
        else:
            model = CNNTransferLearning(best_trial, n_classes=5, device=device, fine_tuning=True).to(device)

    model.load_state_dict(state_dict['model'])
    
    model.eval()
    all_attributions = []
    kernelshap = captum.attr.KernelShap(model)

    for i, (images_batch, labels_batch) in enumerate(test_loader):
        print(f"Batch {i+1}")

        original_im = deprocess_image(images_batch.squeeze(0))
        features = torch.tensor(slic(original_im, n_segments=150, compactness=10, start_label=0))

        images_batch_gpu = test_transforms(images_batch).to(device)
        labels_batch_gpu = labels_batch.to(device)
        baselines_batch_gpu = torch.zeros_like(images_batch_gpu).to(device)
        
        with torch.no_grad():
            out_attr = kernelshap.attribute(
                images_batch_gpu,
                target=labels_batch_gpu,
                baselines=baselines_batch_gpu,
                feature_mask=features.unsqueeze(0).to(device),
                n_samples=1000
            ).cpu()
        
        all_attributions.append(out_attr)
        
        del images_batch_gpu, labels_batch_gpu, baselines_batch_gpu, out_attr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_attributions = torch.cat(all_attributions, dim=0)
    results[name] = final_attributions
    print(f"Finished {name}. Final shape: {final_attributions.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

torch.save(results, 'attributions/kernelshap.pth')