from torch.utils.data import DataLoader
from dg_dataset import Dg_dataset, Dg_dataset_all,transforms
import numpy as np

def data_loaders(datapath, pklpath, batch_size, workers):
    '''Load data with dataloader.'''
    dataset_train = Dg_dataset(datapath, pklpath, datatype='train')
    dataset_valid = Dg_dataset(datapath, pklpath, datatype='valid')

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    return loader_train, loader_valid

def data_loaders_all(datapath, pklpath, batch_size, workers):
    '''Load all data with dataloader.'''
    dataset_all = Dg_dataset_all(datapath, pklpath)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_all = DataLoader(
        dataset_all,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    return loader_all