import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self, diff_kwargs=None):
        init_kwargs = self.init_kwargs
        if diff_kwargs is not None:
            init_kwargs.update(diff_kwargs)
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def num_samples(self):
        return len(self.sampler)


class BaseDataLoaderExplicitSplit(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, args, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle
        self.args = args
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)

class DistBaseDataLoaderExplicitSplit(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.train_sampler = DistributedSampler(dataset)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True,
            'sampler': self.train_sampler
        }
        super().__init__(**self.init_kwargs)

class MultiDistBaseDataLoaderExplicitSplit(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, args, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.args = args
        self.train_sampler = DistributedSampler(dataset, num_replicas=self.args.world_size, rank=self.args.rank, drop_last=True)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True,
            'sampler': self.train_sampler
        }
        super().__init__(**self.init_kwargs)

class BaseMultiDataLoader:
    """
    Currently implemented as undersample the bigger dataloaders...
    """
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.batch_size = self.dataloaders[0].batch_size
    def __getitem__(self, item):
        dl_idx = item % len(self.dataloaders)
        return next(iter(self.dataloaders[dl_idx]))

    def __len__(self):
        return min([len(x) for x in self.dataloaders]) * len(self.dataloaders)

    def num_samples(self):
        return sum([len(x.sampler) for x in self.dataloaders])
