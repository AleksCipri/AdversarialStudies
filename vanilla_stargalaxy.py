import argparse
import logging
import time

from ptlib.dataloaders import StarGalaxyDataManager as DataManager
from ptlib.models import SimpleSGConvNet as Model
from ptlib.trainers import VanillaTrainer as Trainer
from ptlib.utils import get_logging_level
from ptlib.utils import log_function_args


parser = argparse.ArgumentParser(description="Star-Galaxy classifier")
required = parser.add_argument_group("required arguments")
optional = parser.add_argument_group("optional arguments")
optional.add_argument('--batch-size', default=32, type=int, help='batch size')
optional.add_argument('--ckpt-path', default='ckpt.tar', type=str,
                      help='checkpoint path')
required.add_argument('--data-dir', default=None, type=str, help='data dir')
optional.add_argument('--git-hash', default='no hash', type=str,
                      help='git hash')
optional.add_argument('--log-freq', default=100, type=int,
                      help='logging frequency')
optional.add_argument('--log-level', default='INFO', type=str,
                      help='log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)')
optional.add_argument('--num-epochs', default=100, type=int,
                      help='number of epochs')
optional.add_argument('--short-test', default=False, action='store_true',
                      help='do a short test of the code')
optional.add_argument('--show-progress', default=False, action='store_true',
                      help='print tdqm and other output')
optional.add_argument('--test', default=False, action='store_true',
                      help='run on the test set')
optional.add_argument('--tnsrbrd-out-dir', default='/tmp/fashion/tnsrbrd',
                      type=str, help='tensorboardX output dir')
optional.add_argument('--train', default=False, action='store_true',
                      help='do training')


def main(
    batch_size, ckpt_path, data_dir, git_hash, log_freq, log_level, num_epochs,
    short_test, show_progress, test, tnsrbrd_out_dir, train
):
    logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] \
        + str(int(time.time())) + '.txt'
    print('logging to: {}'.format(logfilename))
    logging.basicConfig(
        filename=logfilename, level=get_logging_level(log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Starting...")
    LOGGER.info(__file__)
    log_function_args(vars())

    # set up a data manager and a model
    data_manager = DataManager(data_dir=data_dir)
    data_manager.make_means()
    model = Model()

    # create a trainer with data handler and model
    trainer = Trainer(
        data_manager, model, ckpt_path, tnsrbrd_out_dir,
        log_freq=log_freq, show_progress=show_progress)
    trainer.restore_model_and_optimizer()

    # run training
    if train:
        trainer.train(num_epochs, batch_size, short_test)

    if test:
        _, _, _ = trainer.test(batch_size, short_test)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
