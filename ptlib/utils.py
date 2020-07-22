import logging
import inspect
import json

LOGGER = logging.getLogger(__name__)


def log_function_args(vs):
    '''log a list of vars in a nice json format'''
    LOGGER.info('Calling {}'.format(inspect.stack()[1][3]))
    LOGGER.info(json.dumps(
        vs, indent=3, skipkeys=True, default=repr, sort_keys=True
    ))


def get_logging_level(log_level):
    log_level = log_level.upper()
    logging_level = logging.INFO
    if log_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif log_level == 'INFO':
        logging_level = logging.INFO
    elif log_level == 'WARNING':
        logging_level = logging.WARNING
    elif log_level == 'ERROR':
        logging_level = logging.ERROR
    elif log_level == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        print('Unknown or unset logging level. Using INFO')
    return logging_level


def count_parameters(model):
    '''
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
