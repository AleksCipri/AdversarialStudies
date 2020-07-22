'''
Usage:
    python test_models.py -v
    python test_models.py
'''
import unittest
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import tests.utils as utils

VERBOSE = False
try:
    VERBOSE = bool(os.environ['VERBOSE'])
except KeyError:
    pass


class TestBasicModels(unittest.TestCase):

    def _generate_random_image_and_label(self):
        image_ = np.random.randn(1, 1, 28, 28)
        self.random_image = Variable(
            torch.from_numpy(image_).float()).to(self.device)
        y_ = np.random.randint(10, size=(1))
        self.random_label = \
            torch.from_numpy(y_).type(torch.LongTensor).to(self.device)

    def setUp(self):
        self.model, self.device = utils.configure_and_get_SimpleConvNet()
        self._generate_random_image_and_label()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)

    def test_output_shape(self):
        output = self.model(self.random_image)
        self.assertEqual(
            output.shape, (1, 10), msg="Model output shape is incorrect.")

    def test_conv_output_stats(self):
        # NOTE -- 0.5 is due to the "relu shift"; could change if using a
        # leaky relu or different non-linearity.
        x = F.relu(self.model.conv1(self.random_image))
        self.assertTrue(math.isclose(
            x.data.mean(), 0.5, rel_tol=0.5, abs_tol=0.5),
            msg="Layer 1 conv output mean = {} is not close to 0.5".format(
                x.data.mean()))
        self.assertTrue(math.isclose(
            x.data.std(), 1.0, rel_tol=1.0, abs_tol=1.0),
            msg="Layer 1 conv output std = {} is not close to 1".format(
                x.data.std()))
        if VERBOSE:
            print('conv1(img).mean', x.data.mean())
            print('conv1(img).std', x.data.std())
        x = F.relu(self.model.conv2(x))
        self.assertTrue(math.isclose(
            x.data.mean(), 0.5, rel_tol=0.1, abs_tol=0.1),
            msg="Layer 2 conv output mean = {} is not close to 0.5".format(
                x.data.mean()))
        self.assertTrue(math.isclose(
            x.data.std(), 1.0, rel_tol=1.0, abs_tol=1.0),
            msg="Layer 2 conv output std = {} is not close to 1".format(
                x.data.std()))
        if VERBOSE:
            print('conv2(conv1(img)).mean', x.data.mean())
            print('conv2(conv1(img)).std', x.data.std())
            print(self.model.fc1.weight.data.mean())
            print(self.model.fc1.weight.data.std())

    def test_gradients(self):
        self.model.conv1.weight.retain_grad()
        self.model.conv2.weight.retain_grad()
        for i in range(10):
            if VERBOSE:
                print(' test gradients iter', i)
            self._generate_random_image_and_label()
            self.optimizer.zero_grad()
            output = self.model(self.random_image)
            loss = self.criterion(output, self.random_label)
            if VERBOSE:
                print(loss)
            loss.backward()
            self.optimizer.step()
            self.assertTrue(torch.abs(self.model.conv1.weight.grad.mean()) > 0)
            self.assertTrue(torch.abs(self.model.conv2.weight.grad.mean()) > 0)
            if VERBOSE:
                print(self.model.conv1.weight.grad.mean())
                print(self.model.conv2.weight.grad.mean())

    def test_image_weights_and_activations(self):
        # NOTE - hooks in nn.Modules are super confusing. Especially tricky
        # are checking activations inside an nn.Sequential. See, e.g.,
        # Paperspace blog for more info.
        visualization = {}

        def hook_fn(m, i, o):
            visualization[m] = o

        def get_all_layers(model):
            for name, layer in model._modules.items():
                # don't register on nn.Sequential -- instead recursively
                # register hooks on module children
                if isinstance(layer, nn.Sequential):
                    get_all_layers(layer)
                else:
                    layer.register_forward_hook(hook_fn)

        get_all_layers(self.model)
        _ = self.model(self.random_image)
        if VERBOSE:
            print("\n", visualization.keys())
        for mod, act in visualization.items():
            if isinstance(mod, nn.Conv2d):
                if VERBOSE:
                    print('weight shape = {}, activations shape = {}'.format(
                        mod.weight.shape, act.shape))
                # weight outchannels == activation num filters
                self.assertEqual(mod.weight.shape[0], act.shape[1])
        # TODO -- add some visualization code and save the pdfs; loop over
        # filters in activations and save images, can do same with weights also
        # but vis is harder for 4d tensors, LOL -- so maybe just check stats.


if __name__ == '__main__':
    unittest.main()
