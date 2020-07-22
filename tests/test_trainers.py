'''
Usage:
    python test_trainers.py -v
    python test_trainers.py
'''
import unittest

import ptlib.trainers as trainers
import tests.utils as utils


class TestVanillaTrainer(unittest.TestCase):

    def setUp(self):
        dm = utils.configure_and_get_fash_data_manager()
        model, _ = utils.configure_and_get_SimpleConvNet()
        ckpt_path = './test_vanilla_trainer.tar'
        tnsrboard_out = '/tmp/fashion/test_trainers/'
        self.trainer = trainers.VanillaTrainer(
            dm, model, ckpt_path, tnsrboard_out, log_freq=5)

    def test_private_functions(self):
        batch_idx = self.trainer.log_freq - 1
        self.assertEqual(True, self.trainer._write_to_log(batch_idx))
        self.assertEqual(self.trainer.start_epoch, 0)
        self.trainer._save_state(10)
        self.trainer.restore_model_and_optimizer()
        self.assertEqual(self.trainer.start_epoch, 11)

    def test_run_inference(self):
        batch_size = utils.SYNTH_NUM_SAMPLES // 10
        _, _, test_dl = self.trainer.dm.get_data_loaders(batch_size=batch_size)
        test_loss, correct, total = self.trainer.run_inference_on_dataloader(
            test_dl, short_test=True)
        self.assertIsNotNone(test_loss)
        self.assertIsNotNone(correct)
        self.assertIsNotNone(total)

    def test_train_and_test(self):
        batch_size = utils.SYNTH_NUM_SAMPLES // 10
        self.trainer.train(num_epochs=1, batch_size=batch_size)
        test_loss, correct, total = self.trainer.test(batch_size=batch_size)
        self.assertIsNotNone(test_loss)
        self.assertIsNotNone(correct)
        self.assertIsNotNone(total)


if __name__ == '__main__':
    unittest.main()
