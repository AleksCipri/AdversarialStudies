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
from mathlib.differential_evolution import differential_evolution


LOGGER = logging.getLogger(__name__)


class DiffEvoAttacker(ModelHandlerBase):
    def __init__(self, data_manager, model, ckpt_path, log_freq=100,
                 attack_correct_labels_only=False,
                 hdf5filename_base='diffevo_'):
        '''
        Differential Evolution Attacker ("one pixel" attack)

        By default we attack all images whether they were correctly predicted
        by the model or not. `hdf5filename_base` should be the full path to the
        output HDF5 file for attacked images.
        '''
        super(DiffEvoAttacker, self).__init__(
            data_manager, model, ckpt_path, log_freq)
        self.hdf5filename_base = hdf5filename_base
        self.attack_correct_labels_only = attack_correct_labels_only

    def _de_attack(self, maxiter=75, pixels=1, popsize=400, imgw=48, imgh=48):
        '''
        img width, img height default to 48, 48 -- for star-galaxy dataset;
        note that this is really hardcoded at this point despite implied
        flexibility in these values showing up in the method args...
        TODO - should ask the model how big the input image is
        '''

        def _predict_fn():
            pass

        def _callback_fn():
            pass

        # TODO - color channels hardcoded to CIFAR10 images
        bounds = [(0, imgw), (0, imgh), (0, 255), (0, 255), (0, 255)] * pixels

        popmul = max(1, popsize // len(bounds))

        inits = np.zeros([popmul*len(bounds), len(bounds)])
        for init in inits:
            for i in range(pixels):
                init[i*5+0] = np.random.random()*32
                init[i*5+1] = np.random.random()*32
                init[i*5+2] = np.random.normal(128, 127)
                init[i*5+3] = np.random.normal(128, 127)
                init[i*5+4] = np.random.normal(128, 127)

        attack_result = differential_evolution(
            _predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=_callback_fn, polish=False,
            init=inits)

        print(attack_result)

        # if successful: return 1, attack_result
        # -- 0 is a placeholder...
        return 0, attack_result

    def attack_all(self, max_examps=10, max_iterations=100, num_pixels=1,
                   pop_size=400, short_test=False, targeted=False,
                   verbose=False):
        '''
        Note - this function isn't set up for targeted attakcs yet as we are
        starting with the star-galaxy problem (so, just two classes).
        '''
        LOGGER.info('attack_all')
        if targeted:
            raise ValueError("Not configured for targeted attacks yet.")
        _, _, test_dl = self.dm.get_data_loaders(batch_size=1)
        seen, correct = 0, 0
        true_labels = []
        initial_outputs = []
        perturbed_outputs = []
        adv_example_images = []
        self.model.eval()

        for batch_idx, (inputs, labels) in enumerate(test_dl, 0):
            if short_test and batch_idx >= 2:
                break
            seen += 1

            if max_examps and batch_idx >= max_examps:
                print("stopping after set number of max examples")
                break

            if verbose:
                print("attacking batch_idx = {}".format(batch_idx))
                print("iputs shape:", inputs.shape)
                print("labels:", labels)

            with torch.no_grad():
                output = self.model(inputs)
                if verbose:
                    print(output)
                initial_pred = output.max(1, keepdim=True)[1][0]
                LOGGER.debug('initial pred = {}, label = {}'.format(
                    initial_pred.item(), labels.item()))
                if self.attack_correct_labels_only and \
                        (initial_pred.item() != labels.item()):
                    continue
                correct += 1

#                success, result = self._de_attack(
#                    maxiter=75, pixels=1, popsize=400)


class FGSMAttacker(ModelHandlerBase):
    def __init__(self, data_manager, model, ckpt_path, log_freq=100,
                 attack_correct_labels_only=False, hdf5filename_base='fgsm_'):
        '''
        Fast Gradient Sign Method Attacker

        By default we attack all images whether they were correctly predicted
        by the model or not. `hdf5filename_base` should be the full path to the
        output HDF5 file for attacked images.
        '''
        super(FGSMAttacker, self).__init__(
            data_manager, model, ckpt_path, log_freq)
        self.hdf5filename_base = hdf5filename_base
        self.attack_correct_labels_only = attack_correct_labels_only

    def _fgsm_attack(self, image, epsilon, data_grad):
        '''
        perturbed image = image + epsilon * sign(data_grad)
        '''
        mean = torch.FloatTensor(np.load(self.dm.meanfile))
        std = torch.FloatTensor(np.load(self.dm.stdfile))
        perturbed_image = image * std + mean
        data_grad_sign = data_grad.sign()
        perturbed_image = perturbed_image + epsilon * data_grad_sign
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = (perturbed_image - mean) / std
        # TODO: decide if we need to clamp the image to be in [0, 1]?
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
            inputs.requires_grad = True
            output = self.model(inputs)
            initial_pred = output.max(1, keepdim=True)[1][0]
            LOGGER.debug('initial pred = {}, label = {}'.format(
                initial_pred.item(), labels.item()))
            if self.attack_correct_labels_only and \
                    (initial_pred.item() != labels.item()):
                continue
            seen += 1
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
