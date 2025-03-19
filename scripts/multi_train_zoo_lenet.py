import os
import itertools
import numpy as np
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
    config["model::type"] = "LeNet5"
    config["model::channels_in"] = 3
    config["model::nlin"] = "relu"
    config["model::dropout"] = 0.
    config["model::init_type"] = tune.grid_search(["uniform", "normal", "kaiming_uniform", "kaiming_normal"])
    config["optim::optimizer"] = "adam"
    config["optim::lr"] = tune.grid_search([1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3])
    config["optim::wd"] = tune.grid_search([1e-4, 5e-4, 1e-3])

    config["seed"] = tune.grid_search(range(8))

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

    data_path = "/netscratch2/lmeynent/research/structure_vs_behaviour/results/data/{}_train_zoo_relu/dataset.pt"

    config["dataset::dump"] = tune.grid_search([data_path.format(ds) for ds in ["svhn", "cifar10", "eurosat"]])

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
