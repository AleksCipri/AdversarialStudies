import logging
import torch
import torch.nn as nn
import torch.optim as optim

LOGGER = logging.getLogger(__name__)


class ModelHandlerBase(object):
    def __init__(self, data_manager, model, ckpt_path, log_freq):
        self.dm = data_manager
        self.model = model
        self.ckpt_path = ckpt_path
        self.log_freq = log_freq
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        # TODO - add method to configure criterion and optimizer.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)

    def _write_to_log(self, batch_idx):
        return True if (batch_idx + 1) % self.log_freq == 0 else False

    def _save_state(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.ckpt_path)

    def restore_model_and_optimizer(self):
        try:
            checkpoint = torch.load(self.ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            LOGGER.info('Loaded checkpoint from {}'.format(self.ckpt_path))
        except FileNotFoundError:
            LOGGER.info('No checkpoint found...')

        LOGGER.debug('Model state dict:')
        for param_tensor in self.model.state_dict():
            LOGGER.debug(str(param_tensor) + '\t'
                         + str(self.model.state_dict()[param_tensor].size()))
        LOGGER.debug('Optimizer state dict:')
        for var_name in self.optimizer.state_dict():
            LOGGER.debug(str(var_name) + '\t'
                         + str(self.optimizer.state_dict()[var_name]))

        self.model.to(self.device)

    def run_inference_on_dataloader(
            self, dl, record_results=False, short_test=False):
        # TODO - configure a mechanism for recording inference data to a file.
        correct = 0
        total = 0
        scalar_loss = 0
        with torch.no_grad():
            self.model.eval()
            for i, (images, labels) in enumerate(dl, 0):
                if short_test and i >= 20:
                    break
                images, labels = \
                    images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                # TODO - block for `if record_results` -> record inferences
                loss = self.criterion(outputs, labels)
                scalar_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                if self._write_to_log(i):
                    LOGGER.info(
                        'batch {} truth: '.format(i)
                        + ' '.join('%5s' % self.dm.label_names[labels[j]]
                                   for j in range(4)))
                    LOGGER.info(
                        '         preds: '
                        + ' '.join('%5s' % self.dm.label_names[preds[j]]
                                   for j in range(4)))

        return scalar_loss, correct, total
