from typing import Iterator

# FFCV
from ffcv.loader import Loader

# Scaling Hyper-Reps
from shrp.datasets.augmentations import AugmentationPipeline


class AugmentationDataloader(Iterator):
    def __init__(self, augmentation_pipeline: AugmentationPipeline, *args, **kwargs):
        self.original_dataloader = Loader(*args, **kwargs)
        self.augmentation_pipeline = augmentation_pipeline

    def __iter__(self):
        self.original_dataloader_it = self.original_dataloader.__iter__()
        self.len = self.original_dataloader.__len__()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        next_batch = next(self.original_dataloader_it)
        return self.augmentation_pipeline(*next_batch)
