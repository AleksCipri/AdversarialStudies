'''
FGSM attack code strongly inspired by PyTorch docs example:
    https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
'''
import os
import logging
import h5py
import numpy as np
import torch
from ptlib.model_handlers import ModelHandlerBase

LOGGER = logging.getLogger(__name__)


class FGSMAttacker(ModelHandlerBase):
    def __init__(self, data_manager, model, ckpt_path, log_freq=100):
        '''
        epsilons - scalar or list (iterable) of scales to apply to attack
        '''
        super(FGSMAttacker, self).__init__(
            data_manager, model, ckpt_path, log_freq)
        self.hdf5filename_base = 'fgsm_'
        self.attack_correct_labels_only = False  # TODO - make an init arg

    def _fgsm_attack(self, image, epsilon, data_grad):
        '''
        perturbed image = image + epsilon * sign(data_grad)
        '''
        mean = torch.FloatTensor(np.load(self.dm.meanfile))
        std = torch.FloatTensor(np.load(self.dm.stdfile))
        # TODO - do we need to de-and-re-normalize the image? (we are)
        perturbed_image = image * std + mean
        data_grad_sign = data_grad.sign()
        perturbed_image = perturbed_image + epsilon * data_grad_sign
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = (perturbed_image - mean) / std
        # perturbed_image = torch.clamp(perturbed_image, minval, maxval)
        return perturbed_image

    def _write_adversarial_output(
            self, epsilon, labels,
            init_outputs, perturbed_outputs, adv_examples):
        mean = np.load(self.dm.meanfile)
        std = np.load(self.dm.stdfile)
        labels = torch.stack(labels).numpy()
        init_outputs = torch.stack(init_outputs).numpy()
        perturbed_outputs = torch.stack(perturbed_outputs).numpy()
        adv_examples = torch.stack(adv_examples).numpy() * std + mean
        adv_examples = np.clip(adv_examples, 0, 1)
        # TODO - need an output path for the hdf5s
        hdf5filename = self.hdf5filename_base + \
            '{:4.3f}'.format(epsilon).replace('.', '_') + '.hdf5'
        if os.path.isfile(hdf5filename):
            os.remove(hdf5filename)
        f = h5py.File(hdf5filename, 'w')
        f.create_dataset('catalog', labels.shape,
                         dtype=labels.dtype,
                         compression='gzip')[...] = labels
        f.create_dataset('init_outputs', init_outputs.shape,
                         dtype=init_outputs.dtype,
                         compression='gzip')[...] = init_outputs
        f.create_dataset('perturbed_outputs', perturbed_outputs.shape,
                         dtype=perturbed_outputs.dtype,
                         compression='gzip')[...] = perturbed_outputs
        f.create_dataset('imageset', adv_examples.shape,
                         dtype=adv_examples.dtype,
                         compression='gzip')[...] = adv_examples
        f.close()

    def attack_for_single_epsilon(self, epsilon, short_test=False):
        LOGGER.info(
            'attack_for_single_epsilon for eps = {}'.format(epsilon))
        _, _, test_dl = self.dm.get_data_loaders(batch_size=1)
        seen, correct = 0, 0
        true_labels = []
        initial_outputs = []
        perturbed_outputs = []
        adv_example_images = []
        self.model.eval()

        for iter_num, (inputs, labels) in enumerate(test_dl, 0):
            if short_test and iter_num >= 40:
                break
            seen += 1
            inputs.requires_grad = True
            output = self.model(inputs)
            initial_pred = output.max(1, keepdim=True)[1][0]
            LOGGER.debug('initial pred = {}, label = {}'.format(
                initial_pred.item(), labels.item()))
            # TODO - not sure about this, want to attack wrong labels also?
            if self.attack_correct_labels_only and \
                    (initial_pred.item() != labels.item()):
                continue
            loss = self.criterion(output, labels)
            LOGGER.debug('loss = {}'.format(loss.item()))
            self.model.zero_grad()
            loss.backward()
            inputs_grad = inputs.grad.data
            perturbed_inputs = self._fgsm_attack(inputs, epsilon, inputs_grad)
            perturbed_output = self.model(perturbed_inputs)
            final_pred = perturbed_output.max(1, keepdim=True)[1][0]
            LOGGER.debug('final pred = {}, label = {}'.format(
                final_pred.item(), labels.item()))
            if final_pred.item() == labels.item():
                correct += 1
            true_labels.append(labels.squeeze().detach().cpu())
            initial_outputs.append(
                output.squeeze().detach().cpu())
            perturbed_outputs.append(
                perturbed_output.squeeze().detach().cpu())
            adv_example_images.append(
                perturbed_inputs.squeeze().detach().cpu())

        self._write_adversarial_output(
            epsilon, true_labels, initial_outputs, perturbed_outputs,
            adv_example_images)
        final_accuracy = correct / float(seen)
        LOGGER.info('epsilon: {}, accuracy = {}/{} = {}'.format(
            epsilon, correct, seen, final_accuracy))
