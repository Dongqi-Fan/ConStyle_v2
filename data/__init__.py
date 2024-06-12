import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial

from data.prefetch_dataloader import PrefetchDataLoader
from utils import get_root_logger
from utils.dist_util import get_dist_info
from torchvision import datasets, transforms


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset_type = dataset_opt['type']

    if dataset_type in ['Dataset_Paired']:
        from data.image_datasets import Dataset_Paired as D

    elif dataset_type in ['Dataset_Non_Paired']:
        from data.image_datasets import Dataset_Non_Paired as D

    elif dataset_type in ['Dataset_GaussianDenoising']:
        from data.image_datasets import Dataset_GaussianDenoising as D

    elif dataset_type in ['Dataset_train_ImageNet']:
        from data.image_datasets import Dataset_ImageNet
        logger = get_root_logger()
        logger.info(f'Dataset [{dataset_type}] - {dataset_opt["name"]} is built.')
        # return Dataset_train_ImageNet(dataset_opt)
        return Dataset_ImageNet(dataset_opt, True)

    elif dataset_type in ['Dataset_val_ImageNet']:
        from data.image_datasets import Dataset_ImageNet
        logger = get_root_logger()
        logger.info(f'Dataset [{dataset_type}] - {dataset_opt["name"]} is built.')
        # return Dataset_val_ImageNet(dataset_opt)
        return Dataset_ImageNet(dataset_opt, False)

    elif dataset_type in ['Dataset_JPEG']:
        from data.image_datasets import Dataset_JPEG as D

    elif dataset_type in ['RealESRGANDataset']:
        from data.image_datasets import RealESRGANDataset as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    logger = get_root_logger()
    datasets = D(dataset_opt)
    logger.info(f'Dataset [{datasets.__class__.__name__}] - {dataset_opt["name"]} is built.')
    return datasets


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

