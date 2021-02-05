import argparse
import logging
import time

from ptlib.dataloaders import StarGalaxyDataManager as DataManager
from ptlib.models import SimpleSGConvNet as Model
from ptlib.attackers import DiffEvoAttacker as Attacker
from ptlib.utils import get_logging_level
from ptlib.utils import log_function_args


parser = argparse.ArgumentParser()


parser.add_argument('--ckpt-path', default='ckpt.tar', type=str,
                    help='checkpoint path')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--git-hash', default='no hash', type=str, help='git hash')
parser.add_argument('--log-freq', default=100, type=int,
                    help='logging frequency')
parser.add_argument('--log-level', default='INFO', type=str,
                    help='log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)')
parser.add_argument('--max-examps', default=None, type=int,
                    help='max number of examples to attack')
parser.add_argument('--max-iterations', default=100, type=int,
                    help='max iterations')
parser.add_argument('--num-pixels', default=1, type=int,
                    help='numer of pixels to perturb')
parser.add_argument('--pop-size', default=400, type=int,
                    help='population size')
parser.add_argument('--short-test', default=False, action='store_true',
                    help='do a short test of the code')
parser.add_argument('--targeted', default=False, action='store_true',
                    help='target specific classes')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='print status to the command line')


def main(ckpt_path, data_dir, git_hash, log_freq, log_level, max_examps,
         max_iterations, num_pixels, pop_size, short_test, targeted,
         verbose):
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

    attacker = Attacker(data_manager, model, ckpt_path, log_freq)
    attacker.restore_model_and_optimizer()

    attacker.attack_all(max_examps=max_examps, max_iterations=max_iterations,
                        num_pixels=num_pixels, pop_size=pop_size,
                        short_test=short_test, targeted=targeted,
                        verbose=verbose)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
