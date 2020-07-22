from torch.utils.data import Dataset
import numpy as np
import h5py


class FashionMNISTDataset(Dataset):

    label_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    def __init__(self, hdf5_file, transform=None):
        super(FashionMNISTDataset, self).__init__()
        self._nlabels = 10
        self._file = hdf5_file
        self._f = h5py.File(self._file, 'r')
        self.transform = transform

    def __len__(self):
        return len(self._f['fashion/labels'])

    def __getitem__(self, idx):
        image = self._f['fashion/images'][idx]
        label = self._f['fashion/labels'][idx].reshape([-1])
        oh_label = np.zeros((1, self._nlabels), dtype=np.uint8)
        oh_label[0, label] = 1
        oh_label = oh_label.reshape(self._nlabels,)
        sample = {'image': image, 'label': oh_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class StarGalaxyDataset(Dataset):

    label_names = ['star', 'galaxy']

    def __init__(self, hdf5_file, transform=None):
        super(StarGalaxyDataset, self).__init__()
        self._nlabels = 2
        self._file = hdf5_file
        self._f = h5py.File(self._file, 'r')
        self.transform = transform

    def __len__(self):
        return len(self._f['catalog'])

    def __getitem__(self, idx):
        image = self._f['imageset'][idx]
        label = self._f['catalog'][idx].reshape([-1])
        oh_label = np.zeros((1, self._nlabels), dtype=np.float64)
        oh_label[0, label] = 1
        oh_label = oh_label.reshape(self._nlabels,)
        sample = {'image': image, 'label': oh_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class AttackedStarGalaxyDataset(Dataset):

    label_names = ['star', 'galaxy']

    def __init__(self, hdf5_file, transform=None):
        super(AttackedStarGalaxyDataset, self).__init__()
        self._nlabels = 2
        self._file = hdf5_file
        self._f = h5py.File(self._file, 'r')
        self.transform = transform

    def __len__(self):
        return len(self._f['catalog'])

    def __getitem__(self, idx):
        image = self._f['imageset'][idx]
        label = self._f['catalog'][idx].reshape([-1])
        init_outputs = self._f['init_outputs'][idx]
        perturbed_outputs = self._f['perturbed_outputs'][idx]
        oh_label = np.zeros((1, self._nlabels), dtype=np.float64)
        oh_label[0, label] = 1
        oh_label = oh_label.reshape(self._nlabels,)
        sample = {
            'image': image,
            'label': oh_label,
            'init_outputs': init_outputs,
            'perturbed_outputs': perturbed_outputs}
        if self.transform:
            sample = self.transform(sample)
        return sample
