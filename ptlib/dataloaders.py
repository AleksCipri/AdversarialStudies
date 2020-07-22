import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import h5py

import os

from ptlib.datasets import FashionMNISTDataset
from ptlib.datasets import StarGalaxyDataset
from ptlib.transforms import Standardize
from ptlib.transforms import ToTensor
from ptlib.transforms import AttackedToTensor


class WrapDataLoader(object):
    '''
    class to manage pulling returned dict apart, following project conventions
    '''

    def __init__(self, dl):
        self.dl = dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for data in batches:
            yield data['image'], data['label']


class DataManagerBase(object):
    '''base class for the data managers'''
    def __init__(self, data_dir, data_set_cls):
        self.data_dir = data_dir
        self.data_set_cls = data_set_cls
        self.testfile = None
        self.trainfile = None
        self.validfile = None
        self.meanfile = None
        self.stdfile = None
        self.label_names = None

    def make_means(self):
        for filename in [self.meanfile, self.stdfile]:
            if os.path.isfile(filename):
                os.remove(filename)
        f = h5py.File(self.trainfile, 'r')
        m = np.mean(f[self.img_h5_dset_name], axis=0)
        s = np.std(f[self.img_h5_dset_name], axis=0)
        np.save(self.meanfile, m)
        np.save(self.stdfile, s)
        f.close()

    def get_data_loaders(self, batch_size, standardize=True):
        standardizer = Standardize(
            mean_file=self.meanfile, std_file=self.stdfile)
        trnsfrms = transforms.Compose([
            standardizer, ToTensor()
        ]) if standardize else ToTensor()

        train_dataloader = None
        if self.trainfile is not None:
            trainset = self.data_set_cls(self.trainfile, trnsfrms)
            train_dataloader = WrapDataLoader(DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1
            ))

        valid_dataloader = None
        if self.validfile is not None:
            validset = self.data_set_cls(self.validfile, trnsfrms)
            valid_dataloader = WrapDataLoader(DataLoader(
                validset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1
            ))

        test_dataloader = None
        if self.testfile is not None:
            testset = self.data_set_cls(self.testfile, trnsfrms)
            test_dataloader = WrapDataLoader(DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1
            ))

        return train_dataloader, valid_dataloader, test_dataloader


class FashionDataManager(DataManagerBase):
    '''main data access class for Fashion MNIST'''
    def __init__(self, data_dir):
        super(FashionDataManager, self).__init__(data_dir, FashionMNISTDataset)
        self.testfile = None
        self.trainfile = os.path.join(data_dir, 'fashion_train.hdf5')
        self.validfile = os.path.join(data_dir, 'fashion_test.hdf5')
        self.meanfile = os.path.join(data_dir, 'fashion_mean.npy')
        self.stdfile = os.path.join(data_dir, 'fashion_stddev.npy')
        self.label_names = FashionMNISTDataset.label_names
        self.img_h5_dset_name = 'fashion/images'


class StarGalaxyDataManager(DataManagerBase):
    '''main data access class for Star-Galaxy dset'''
    def __init__(self, data_dir):
        super(StarGalaxyDataManager, self).__init__(
            data_dir, StarGalaxyDataset)
        self.testfile = os.path.join(
            data_dir, 'stargalaxy_real_ptflt_test.hdf5')
        self.trainfile = os.path.join(
            data_dir, 'stargalaxy_real_ptflt_train.hdf5')
        self.validfile = os.path.join(
            data_dir, 'stargalaxy_real_ptflt_valid.hdf5')
        self.meanfile = os.path.join(data_dir, 'star_galaxy_mean.npy')
        self.stdfile = os.path.join(data_dir, 'star_galaxy_stddev.npy')
        self.label_names = StarGalaxyDataset.label_names
        self.img_h5_dset_name = 'imageset'


class WrapAttackedDataLoader(object):
    '''
    class to manage pulling returned dict apart, following project conventions
    '''

    def __init__(self, dl):
        self.dl = dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for data in batches:
            yield data['image'], data['label'], \
                data['init_outputs'], data['perturbed_outputs']


class AttackedDataManager(object):
    '''data manager class for attacked data'''
    def __init__(self, data_full_path, data_set_cls):
        self.data_full_path = data_full_path
        self.data_set_cls = data_set_cls
        self.label_names = data_set_cls.label_names

    def get_data_loader(self, batch_size):
        trnsfrms = AttackedToTensor()

        dataset = self.data_set_cls(self.data_full_path, trnsfrms)
        dataloader = WrapAttackedDataLoader(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
        ))

        return dataloader
