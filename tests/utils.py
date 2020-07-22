import os
import h5py
import numpy as np
import torch

import ptlib.dataloaders as dataloaders
import ptlib.models as models
from ptlib.datasets import AttackedStarGalaxyDataset

SYNTH_NUM_SAMPLES = 100
FASH_TRAINH5 = 'fash_synth_train.h5'
FASH_TESTH5 = 'fash_synth_test.h5'
FASH_VALIDH5 = 'fash_synth_valid.h5'
FASH_MEANFILE = 'fash_synth_mean.npy'
FASH_STDFILE = 'fash_synth_std.npy'
SG_TRAINH5 = 'sg_synth_train.h5'
SG_TESTH5 = 'sg_synth_test.h5'
SG_VALIDH5 = 'sg_synth_valid.h5'
SG_MEANFILE = 'sg_synth_mean.npy'
SG_STDFILE = 'sg_synth_std.npy'
ATK_SG_H5 = 'atk_sg_synth.h5'


def cleanup_fash_synth():
    for filename in [FASH_TRAINH5, FASH_TESTH5, FASH_VALIDH5,
                     FASH_MEANFILE, FASH_STDFILE]:
        if os.path.isfile(filename):
            os.remove(filename)


def cleanup_sg_synth():
    for filename in [SG_TRAINH5, SG_TESTH5, SG_VALIDH5,
                     SG_MEANFILE, SG_STDFILE]:
        if os.path.isfile(filename):
            os.remove(filename)


def cleanup_atk_sg_synth():
    for filename in [ATK_SG_H5]:
        if os.path.isfile(filename):
            os.remove(filename)


def make_fash_h5():
    cleanup_fash_synth()
    synth_train_images = np.random.randint(
        0, high=256, size=(SYNTH_NUM_SAMPLES, 1, 28, 28)).astype('uint8')
    synth_train_labels = np.random.randint(
        0, high=10, size=(SYNTH_NUM_SAMPLES, 1)).astype('uint8')
    synth_test_images = np.random.randint(
        0, high=256, size=(SYNTH_NUM_SAMPLES, 1, 28, 28)).astype('uint8')
    synth_test_labels = np.random.randint(
        0, high=10, size=(SYNTH_NUM_SAMPLES, 1)).astype('uint8')
    fill_fash_hdf5(FASH_TESTH5, synth_test_images, synth_test_labels)
    fill_fash_hdf5(FASH_TRAINH5, synth_train_images, synth_train_labels)
    fill_fash_hdf5(FASH_VALIDH5, synth_test_images, synth_test_labels)


def make_sg_h5():
    cleanup_sg_synth()
    synth_train_images = np.random.rand(SYNTH_NUM_SAMPLES, 3, 48, 48)
    synth_train_labels = np.random.randint(
        0, high=2, size=(SYNTH_NUM_SAMPLES, 1)).astype('uint8')
    synth_test_images = np.random.rand(SYNTH_NUM_SAMPLES, 3, 48, 48)
    synth_test_labels = np.random.randint(
        0, high=2, size=(SYNTH_NUM_SAMPLES, 1)).astype('uint8')
    fill_sg_hdf5(SG_TESTH5, synth_test_images, synth_test_labels)
    fill_sg_hdf5(SG_TRAINH5, synth_train_images, synth_train_labels)
    fill_sg_hdf5(SG_VALIDH5, synth_test_images, synth_test_labels)


def make_atk_sg_h5():
    cleanup_atk_sg_synth()
    synth_images = np.random.rand(SYNTH_NUM_SAMPLES, 3, 48, 48)
    synth_labels = np.random.randint(
        0, high=2, size=(SYNTH_NUM_SAMPLES, 1)).astype('uint8')
    synth_init_out = np.random.rand(SYNTH_NUM_SAMPLES, 2)
    synth_pert_out = np.random.rand(SYNTH_NUM_SAMPLES, 2)
    fill_atk_sg_hdf5(ATK_SG_H5, synth_images, synth_labels,
                     synth_init_out, synth_pert_out)


def fill_fash_hdf5(file_name, images, labels):
    f = h5py.File(file_name, 'w')
    grp = f.create_group('fashion')
    images_set = grp.create_dataset(
        'images', np.shape(images), dtype='uint8', compression='gzip'
    )
    labels_set = grp.create_dataset(
        'labels', np.shape(labels), dtype='uint8', compression='gzip'
    )
    images_set[...] = images
    labels_set[...] = labels
    f.close()


def fill_sg_hdf5(file_name, images, labels):
    f = h5py.File(file_name, 'w')
    f.create_dataset(
        'imageset', np.shape(images), dtype='float64', compression='gzip'
    )[...] = images
    f.create_dataset(
        'catalog', np.shape(labels), dtype='uint8', compression='gzip'
    )[...] = labels
    f.close()


def fill_atk_sg_hdf5(file_name, images, labels, init_out, pert_out):
    f = h5py.File(file_name, 'w')
    f.create_dataset(
        'imageset', np.shape(images), dtype='float64', compression='gzip'
    )[...] = images
    f.create_dataset(
        'init_outputs', np.shape(init_out), dtype='float64', compression='gzip'
    )[...] = init_out
    f.create_dataset(
        'perturbed_outputs', np.shape(pert_out), dtype='float64',
        compression='gzip'
    )[...] = pert_out
    f.create_dataset(
        'catalog', np.shape(labels), dtype='uint8', compression='gzip'
    )[...] = labels
    f.close()


def configure_and_get_fash_data_manager():
    dm = dataloaders.FashionDataManager(data_dir=".")
    make_fash_h5()
    dm.testfile = FASH_TESTH5
    dm.trainfile = FASH_TRAINH5
    dm.validfile = FASH_VALIDH5
    dm.meanfile = FASH_MEANFILE
    dm.stdfile = FASH_STDFILE
    dm.make_means()
    return dm


def configure_and_get_sg_data_manager():
    dm = dataloaders.StarGalaxyDataManager(data_dir=".")
    make_sg_h5()
    dm.testfile = SG_TESTH5
    dm.trainfile = SG_TRAINH5
    dm.validfile = SG_VALIDH5
    dm.meanfile = SG_MEANFILE
    dm.stdfile = SG_STDFILE
    dm.make_means()
    return dm


def configure_and_get_atk_sg_data_manager():
    make_sg_h5()
    dm = dataloaders.AttackedDataManager(
        data_full_path=ATK_SG_H5, data_set_cls=AttackedStarGalaxyDataset)
    return dm


def configure_and_get_SimpleConvNet():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.SimpleConvNet()
    model.to(device)
    return model, device
