import logging

logging.basicConfig(level=logging.INFO)

import os
import numpy as np

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import torch

import ray
from ray import tune

# from ray.air.integrations.wandb import WandbLoggerCallback
from shrp.evaluation.ray_fine_tuning_callback import CheckpointSamplingCallback

import json

from pathlib import Path


from shrp.models.def_AE_trainable import AE_trainable
from shrp.datasets.dataset_tokens import DatasetTokens

from shrp.datasets.dataset_ffcv import prepare_ffcv_dataset

from shrp.git_re_basin.git_re_basin import (
    zoo_cnn_large_permutation_spec
)


PATH_ROOT = Path("/netscratch2/lmeynent/research/structure_vs_behaviour/results/hyperrepresentations")
PATH_ZOO = Path("/netscratch2/lmeynent/research/structure_vs_behaviour/results/zoos/cifar10_train_zoo_relu")

def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    cpus_per_trial = 8
    gpus_per_trial = 1
    gpus = 3
    cpus = 32

    # round down to maximize GPU usage

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    print(f"resources_per_trial: {resources_per_trial}")

    ### configure experiment #########
    experiment_name = Path(__file__).stem
    # set module parameters
    config = {}
    config["seed"] = 2020
    config["device"] = "cuda"
    config["device_no"] = 0
    config["training::precision"] = "amp"
    # config["trainset::precision"] = "16"
    # config["trainset::batchsize"] = 64
    config["trainset::batchsize"] = 64

    # permutation specs
    config["training::permutation_number"] = 20
    config["training::view_1_canon"] = False
    config["training::view_2_canon"] = False

    config["testing::permutation_number"] = 20
    config["testing::view_1_canon"] = False
    config["testing::view_2_canon"] = False
    # config["testing::permutations_per_sample"] = 5

    config["training::reduction"] = "mean"

    config["ae:i_dim"] = 289
    config["ae:lat_dim"] = 64
    # config["ae:lat_dim"] = 64
    # 9609,   40,  511
    config["ae:max_positions"] = [500, 50, 100]
    # config["training::windowsize"] = tune.grid_search([2028, 1536, 1024, 512])
    config["ae:d_model"] = 256
    # config["ae:nhead"] = tune.grid_search([4, 8, 16])
    config["ae:nhead"] = 8
    # config["ae:num_layers"] = tune.grid_search([4, 8, 16])
    config["ae:num_layers"] = 8

    # configure optimizer
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = tune.grid_search([1e-5, 5e-5, 1e-4])
    config["optim::wd"] = 3e-9

    # training config
    config["training::temperature"] = 0.1
    # config["training::gamma"] = 1.0
    config["training::gamma"] = 0.05
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"
    # AMP
    #
    config["training::epochs_train"] = 100
    config["training::output_epoch"] = 10
    # config["training::output_epoch"] = 1
    config["training::test_epochs"] = 5

    ### Distillation loss
    for path in os.listdir(PATH_ZOO):
        if 'NN_tune' in path and "_00000_0" in path:
            ref_path = PATH_ZOO / path
    config["training::distil_reference"] = ref_path
    
    config["training::beta"] = tune.grid_search([0., 0.1, 0.25, 0.5, 0.75, 1.])
    config["training::loss_distillation"] = tune.grid_search(['l2', 'distillation'])
    config["training::temperature_distillation"] = 2.0
    config["training::queryset_distillation"] = "data"
    config["training::queryset_dump"] = "/netscratch2/lmeynent/research/structure_vs_behaviour/results/data/cifar10_train_zoo_relu/dataset.pt"
    config["training::n_queries_distillation"] = 256

    # configure output path
    output_dir = PATH_ROOT.joinpath("tune")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass

    ###### Datasets ###########################################################################
    # pre-compute dataset and drop in torch.save
    # data_path = output_dir.joinpath(experiment_name)
    # data_path = Path("/raid/kschuerholt/dataset_mnist_cnn_std")
    data_path = PATH_ZOO
    data_path.mkdir(exist_ok=True)

    result_key_list = ["test_acc", "training_iteration", "ggap"]
    config_key_list = [
        "model::init_type",
        "optim::lr",
        "optim::wd",
    ]
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    dataset = dict()

    dataset['trainset'] = DatasetTokens(
        root=PATH_ZOO,
        train_val_test="train",
        max_samples=64,
        standardize=False,
        #tokensize=config["ae:i_dim"],
        property_keys=property_keys
    )

    config["training::windowsize"] = dataset['trainset'][0][0].shape[0]

    # output_dir.joinpath(experiment_name).mkdir(exist_ok=True)
    # path to ffcv dataset for training
    config["dataset::dump"] = data_path.joinpath("dataset_beton").absolute()
    # path to .pt dataset for downstream tasks
    config["downstreamtask::dataset"] =  data_path.joinpath("dataset.pt").absolute()
    # call dataset prepper function
    if not (data_path / "dataset_beton.train").is_file():
        logging.info("prepare data")
        prep_data(target_dataset_path=data_path, config=config, property_keys=property_keys)

    ### Augmentations
    config["trainloader::workers"] = 8
    config["trainset::add_noise_view_1"] = 0.
    config["trainset::add_noise_view_2"] = 0.
    config["trainset::noise_multiplicative"] = None
    config["trainset::erase_augment_view_1"] = None
    config["trainset::erase_augment_view_2"] = None

    config["callbacks"] = []

    config["resources"] = resources_per_trial
    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
    )
    assert ray.is_initialized() == True

    experiment = ray.tune.Experiment(
        name=experiment_name,
        run=AE_trainable,
        stop={
            "training_iteration": config["training::epochs_train"],
        },
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=output_dir,
        resources_per_trial=resources_per_trial,
    )
    # run
    ray.tune.run_experiments(
        experiments=experiment,
        # resume="ERRORED_ONLY", # resumes from previous run. if run should be done all over, set resume=False
        resume=False,
        # resume=True,  # resumes from previous run. if run should be done all over, set resume=False
        reuse_actors=False,
        callbacks=[],
        verbose=3,
    )

    ray.shutdown()
    assert ray.is_initialized() == False


####################################################################################
# prepare data
def prep_data(target_dataset_path, config, property_keys):
    dataset_target_path = target_dataset_path

    permutation_spec = zoo_cnn_large_permutation_spec()
    map_to_canonical = False
    standardize = False
    weight_threshold = 10_000
    num_threads = 16
    shuffle_path = True
    # windowsize = 1024 + 512
    windowsize = config["training::windowsize"]
    # supersample = "auto"
    # supersample = 100
    # supersample = "auto"
    supersample = 1
    precision = "32"
    # precision = "b16"
    # ignore_bn = True
    ignore_bn = False
    tokensize = 0

    drop_pt_dataset = True

    # permutation spec
    permutation_number_train = config["training::permutation_number"]
    permutations_per_sample_train = max(5, permutation_number_train)
    permutation_number_test = config["testing::permutation_number"]
    permutations_per_sample_test = max(5, permutation_number_test)

    page_size = 2**27
    # splits = ["train"]
    # zoo_path = [Path("/ds2/model_zoos/zoos_backdoors/round2_resnet18/train").absolute()]
    # ds_split = [1.0, 0.0, 0.0]
    # page_size = 2**25
    # splits = ["val"]
    epoch_list = [20, 30, 40, 50]
    zoo_path = [PATH_ZOO.absolute()]
    ds_split = [0.8, 0.05, 0.15]
    max_samples = len([path for path in os.listdir(PATH_ZOO) if 'NN_tune' in path])
    splits = ["train", "val", "test"]
    # zoo_path = [Path("/ds2/model_zoos/zoos_backdoors/round2_resnet18/test").absolute()]
    # ds_split = [0.0, 0.0, 1.0]
    # page_size = 4 * 1 << 21  # (2**23)
    #splits = ['train', 'test']

    prepare_ffcv_dataset(
        dataset_target_path=dataset_target_path,
        zoo_path=zoo_path,
        epoch_list=epoch_list,
        permutation_spec=permutation_spec,
        map_to_canonical=map_to_canonical,
        standardize=standardize,  
        ds_split=ds_split,
        max_samples=max_samples,
        weight_threshold=weight_threshold,
        property_keys=property_keys,
        num_threads=num_threads,
        shuffle_path=shuffle_path,
        windowsize=windowsize,
        supersample=supersample,
        precision=precision,
        splits=splits,
        ignore_bn=ignore_bn,
        tokensize=tokensize,
        permutation_number_train=permutation_number_train,
        permutations_per_sample_train=permutations_per_sample_train,
        permutation_number_test=permutation_number_test,
        permutations_per_sample_test=permutations_per_sample_test,
        page_size=page_size,
        drop_pt_dataset=drop_pt_dataset
    )


if __name__ == "__main__":
    main()
