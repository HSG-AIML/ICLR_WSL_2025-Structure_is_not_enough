import os
import itertools
import numpy as np
from torchvision.datasets import CIFAR10
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
    # ray init to limit memory and storage
    cpus = 32
    gpus = 3

    cpu_per_trial = 1
    gpu_fraction = 0.1
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_fraction}

    # experiment name
    experiment_name = experiment_name = Path(__file__).stem

    # set module parameters
    config = {}
    config["model::type"] = "CNN3"
    config["model::channels_in"] = 3
    config["model::nlin"] = "relu"
    config["model::dropout"] = 0.
    config["model::init_type"] = tune.grid_search(["uniform", "normal", "kaiming_uniform", "kaiming_normal"])
    config["optim::optimizer"] = "adam"
    config["optim::lr"] = tune.grid_search([1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3])
    config["optim::wd"] = tune.grid_search([1e-4, 5e-4, 1e-3])

    config["seed"] = tune.grid_search(range(20))

    # set training parameters
    net_dir = PATH_ROOT / 'zoos'
    try:
        net_dir.joinpath(experiment_name).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    print(f"Zoo directory: {net_dir.absolute()}")

    # config["training::batchsize"] = tune.grid_search([8, 4, 2])
    config["training::batchsize"] = 32
    config["training::epochs_train"] = 50
    config["training::start_epoch"] = 1
    config["training::output_epoch"] = 1
    config["training::val_epochs"] = 1
    config["training::idx_out"] = 500
    config["training::checkpoint_dir"] = None

    config["cuda"] = True if gpus > 0 and torch.cuda.is_available() else False

    data_path = PATH_ROOT / 'data' / experiment_name
    print(f"Data directory: {data_path.absolute()}")
    try:
        net_dir.joinpath(data_path).mkdir(parents=True, exist_ok=True)
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
        
        trainset_raw = CIFAR10(
            root='/netscratch2/lmeynent/datasets/CIFAR10',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            download=True
        )

        testset_raw = CIFAR10(
            root='/netscratch2/lmeynent/datasets/CIFAR10',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
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

    config["dataset::dump"] = str(data_path.joinpath(f"dataset.pt"))

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
    )

    # save config as json file
    with open((net_dir.joinpath(experiment_name, "config.json")), "w") as f:
        json.dump(config, f, default=str)

    # generate empty readme.md file   ?? maybe create and copy template
    # check if readme.md exists
    readme_file = net_dir.joinpath(experiment_name, "readme.md")
    if readme_file.is_file():
        pass
    # if not, make empty readme
    else:
        with open(readme_file, "w") as f:
            pass

    assert ray.is_initialized() == True

    # run tune trainable experiment
    analysis = tune.run(
        NN_tune_trainable,
        name=experiment_name,
        stop={"training_iteration": config["training::epochs_train"], },
        checkpoint_score_attr="test_acc",
        checkpoint_freq=config["training::output_epoch"],
        config=config,
        local_dir=str(net_dir),
        reuse_actors=False,
        # resume="ERRORED_ONLY",  # resumes from previous run. if run should be done all over, set resume=False
        # resume="LOCAL",  # resumes from previous run. if run should be done all over, set resume=False
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        resources_per_trial=resources_per_trial,
        verbose=3
    )

    ray.shutdown()
assert ray.is_initialized() == False


if __name__ == "__main__":
    main()
