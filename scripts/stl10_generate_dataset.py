import os
import itertools
import numpy as np
from torchvision.datasets import STL10
from torchvision import transforms
import torch
from ray import tune
import ray
import json
from pathlib import Path
import sys
sys.path.append('./..')

from shrp.models.def_NN_experiment import NN_tune_trainable


PATH_ROOT = Path("/netscratch2/lmeynent/research/structure_vs_behaviour/results")

def main():
    data_path = PATH_ROOT / 'data' / 'stl10_32'
    print(f"Data directory: {data_path.absolute()}")
    try:
        data_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass


    try:
        # load existing dataset
        dataset = torch.load(str(data_path.joinpath(f"dataset.pt")))

    except FileNotFoundError:
        # if file not found, generate and save dataset
        # seed for reproducibility

        dataset_path = str(data_path.joinpath(f"dataset.pt"))

        dataset_seed = 2020

        transforms_stl10 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        trainset_raw = STL10(
            root='/netscratch2/lmeynent/datasets/STL-10',
            split='train',
            transform=transforms_stl10,
            download=True
        )

        testset_raw = STL10(
            root='/netscratch2/lmeynent/datasets/STL-10',
            split='test',
            transform=transforms_stl10,
            download=True
        )
        
        # split trainset into trainset and valset
        train_frac = 0.9
        train_len = int(len(trainset_raw) * train_frac)

        trainset_raw, valset_raw = torch.utils.data.random_split(
            trainset_raw, [train_len, len(trainset_raw) - train_len], generator=torch.Generator().manual_seed(dataset_seed))

        # temp dataloaders
        trainloader_raw = torch.utils.data.DataLoader(
            dataset=trainset_raw, batch_size=len(trainset_raw), shuffle=True
        )
        valloader_raw = torch.utils.data.DataLoader(
            dataset=valset_raw, batch_size=len(valset_raw), shuffle=True
        )
        testloader_raw = torch.utils.data.DataLoader(
            dataset=testset_raw, batch_size=len(testset_raw), shuffle=True
        )
        # one forward pass
        assert trainloader_raw.__len__() == 1, "temp trainloader has more than one batch"
        for train_data, train_labels in trainloader_raw:
            pass
        assert valloader_raw.__len__() == 1, "temp valloader has more than one batch"
        for val_data, val_labels in valloader_raw:
            pass
        assert testloader_raw.__len__() == 1, "temp testloader has more than one batch"
        for test_data, test_labels in testloader_raw:
            pass

        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        valset = torch.utils.data.TensorDataset(val_data, val_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # save dataset and seed in data directory
        dataset = {
            "trainset": trainset,
            "valset": valset,
            "testset": testset,
            "dataset_seed": dataset_seed
        }
        torch.save(dataset, dataset_path)


if __name__ == "__main__":
    main()
