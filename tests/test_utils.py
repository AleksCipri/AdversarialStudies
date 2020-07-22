'''
Usage:
    python test_utils.py -v
    python test_utils.py
'''
import logging
import unittest
import torch
import ptlib.utils as utils
import ptlib.models as models


class TestUtils(unittest.TestCase):

    def test_get_logging_level(self):
        self.assertEqual(logging.INFO,
                         utils.get_logging_level('info'))
        self.assertEqual(logging.INFO,
                         utils.get_logging_level('INFO'))
        self.assertEqual(logging.DEBUG,
                         utils.get_logging_level('debug'))
        self.assertEqual(logging.DEBUG,
                         utils.get_logging_level('DEBUG'))
        self.assertEqual(logging.WARNING,
                         utils.get_logging_level('warning'))
        self.assertEqual(logging.WARNING,
                         utils.get_logging_level('WARNING'))
        self.assertEqual(logging.ERROR,
                         utils.get_logging_level('error'))
        self.assertEqual(logging.ERROR,
                         utils.get_logging_level('ERROR'))
        self.assertEqual(logging.CRITICAL,
                         utils.get_logging_level('critical'))
        self.assertEqual(logging.CRITICAL,
                         utils.get_logging_level('CRITICAL'))

    def test_parameter_count(self):
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        model = models.SimpleConvNet()
        model.to(device)
        n_params = utils.count_parameters(model)
        self.assertGreater(n_params, 0)
        print("\n---")
        print(model.__class__.__name__, "has", n_params, "params")
        print("---")


if __name__ == '__main__':
    unittest.main()
