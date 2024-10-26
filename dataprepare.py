import pandas as pd
import numpy as np
from torchvision import datasets
import pathlib
import splitfolders
from torchvision.transforms import transforms
import torchvision
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import copy

def load_data(dataset_directory):
    dataset_directory = pathlib.Path(dataset_directory)
    splitfolders.ratio(dataset_directory, output = 'brain', seed = 20,ratio = (0.9,0.05,0.05))

def make_transformations(dataset_directory):
    dataset_directory = pathlib.Path(dataset_directory)
    
    train_formations = transforms.Compose(
        [
         transforms.Resize((256,256)),
         transforms.RandomVerticalFlip(p = 0.5),
         transforms.RandomHorizontalFlip(p = 0.5),
         transforms.RandomRotation(30),
         transforms.ToTensor(),
         transforms.Normalize(mean = [0.485,0.456,0.406], std =[0.229, 0.224, 0.225])
        ]
    )
    
    train_set = torchvision.datasets.ImageFolder(dataset_directory.joinpath('train'), transform=train_formations)
    train_set.transform
    test_set = torchvision.datasets.ImageFolder(dataset_directory.joinpath('val'),transform=train_formations)
    test_set.transform
    datasets = [train_set,test_set]
    return train_set, test_set

def luismi_transformations(dataset_directory):
    dataset_directory = pathlib.Path(dataset_directory)
    
    train_formations = transforms.Compose(
        [
         transforms.Resize((256,256)),
         transforms.RandomVerticalFlip(p = 0.5),
         transforms.RandomHorizontalFlip(p = 0.5),
         transforms.RandomRotation(30),
         transforms.ToTensor(),
         transforms.Normalize(mean = [0.485,0.456,0.406], std =[0.229, 0.224, 0.225])
        ]
    )

    test_transformations = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485,0.456,0.406], std =[0.229, 0.224, 0.225])
    ])
    
    train_set = torchvision.datasets.ImageFolder(dataset_directory.joinpath('train'), transform=train_formations)

    test_set = torchvision.datasets.ImageFolder(dataset_directory.joinpath('test'),transform=test_transformations)

    dev_set = torchvision.datasets.ImageFolder(dataset_directory.joinpath('val'), transform=test_transformations)
    
    return train_set, test_set, dev_set

def data_visualization(train_set, test_set, cols,rows):
    
    labels_dict = {
        0: 'Brain Tumor',
        1: 'Healthy'
    }
    fig = plt.figure(figsize=(12,8))
    for i in range(1,cols*rows + 1):
        data_index = torch.randint(0, len(train_set), size=(1,)).item()
        img,label = train_set[data_index]
        fig.add_subplot(rows,cols, i)
        plt.title(labels_dict[label])
        img_np = img.numpy().transpose(1,2,0)
        img_valid_range = np.clip(img_np,0,1)
        plt.imshow(img_valid_range)
        plt.suptitle('Brain images' , y = 0.95)
    plt.show()


def data_loaders(train_set,test_set,dev_set):
    train_load = DataLoader(train_set, batch_size=64, shuffle=True)
    test_load = DataLoader(test_set, batch_size=64, shuffle=True)
    dev_load = DataLoader(dev_set, batch_size=64, shuffle=True)
    return train_load, test_load, dev_load






