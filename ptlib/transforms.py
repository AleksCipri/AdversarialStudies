import torch
import numpy as np


class Standardize(object):
    '''transform to shift images to unit variance and zero mean'''

    def __init__(self, mean_file, std_file):
        self.mean = np.load(mean_file)
        self.std = np.load(std_file)
        assert (self.std == 0).any() == False, '0-values in std. dev.'

    def __call__(self, sample):
        img = (sample['image'] - self.mean) / self.std
        return {'image': img, 'label': sample['label']}


class ToTensor(object):
    '''transform for moving fashion and stargalaxy data to tensors'''

    def __call__(self, sample):
        '''CrossEntropyLoss does not expect a one-hot encoded vector'''
        image, label = sample['image'], sample['label']
        # torch.max(torch.from_numpy(label).type(torch.LongTensor))
        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.argmax(
                torch.from_numpy(label).type(torch.LongTensor)
            )
        }


class AttackedToTensor(object):
    '''
    transform for moving attacked data to tensors
    '''

    def __call__(self, sample):
        '''CrossEntropyLoss does not expect a one-hot encoded vector'''
        image, label, init_outputs, perturbed_outputs = \
            sample['image'], \
            sample['label'], \
            sample['init_outputs'], \
            sample['perturbed_outputs']
        return {
            'image': torch.from_numpy(image).float(),
            'init_outputs': torch.from_numpy(init_outputs).float(),
            'perturbed_outputs': torch.from_numpy(perturbed_outputs).float(),
            'label': torch.argmax(
                torch.from_numpy(label).type(torch.LongTensor)
            )
        }
