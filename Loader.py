import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader


class Loader:
    def LoadData(train_dataset_path = "", val_dataset_path = "", test_dataset_path = "", batch_size = 16, image_size = 224):
        mean = [0.3141, 0.3803, 0.3729]
        std = [0.2814, 0.3255, 0.3267]

        train_transforms = transforms.Compose([
            transforms.Resize((image_size,image_size)), 
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((image_size,image_size)), 
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((image_size,image_size)), 
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])

        train_loader = None
        test_loader = None
        val_loader = None

        if train_dataset_path != "":
            train_dataset = torchvision.datasets.ImageFolder(root = train_dataset_path, transform = train_transforms)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if test_dataset_path != "":
            test_dataset = torchvision.datasets.ImageFolder(root = test_dataset_path, transform = test_transforms)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        if val_dataset_path != "":
            val_dataset = torchvision.datasets.ImageFolder(root = val_dataset_path, transform = val_transforms)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return  train_loader, val_loader, test_loader