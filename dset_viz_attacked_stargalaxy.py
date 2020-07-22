'''
plot attacked star-galaxy images
'''
import argparse
import numpy as np
import torch
from torch.nn import Softmax
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ptlib.dataloaders import AttackedDataManager as DataManager
from ptlib.datasets import AttackedStarGalaxyDataset as AttackedDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=6, type=int, help='batch size')
parser.add_argument('--data-full-path', default='', type=str,
                    help='data full path')
parser.add_argument('--dtype', default='float64', type=str,
                    help='img data type')
parser.add_argument('--num-batches', default=1, type=int,
                    help='number of batches')
parser.add_argument('--pdf-name', default='evt_all.pdf', type=str,
                    help='output pdf name')


def main(batch_size, data_full_path, dtype, num_batches, pdf_name):
    sm = Softmax(dim=0)
    data_manager = DataManager(data_full_path=data_full_path,
                               data_set_cls=AttackedDataset)
    dl = data_manager.get_data_loader(batch_size=batch_size)
    with PdfPages(pdf_name) as pdf:
        for iter_num, (inputs, labels, init_outputs, perturbed_outputs) \
                in enumerate(dl, 0):
            if iter_num >= num_batches:
                break
            n_cols = int(np.sqrt(batch_size))
            n_rows = int(np.ceil(batch_size / n_cols))
            # make grid, plots
            gs = plt.GridSpec(n_rows, n_cols)
            for evt, img_tnsr in enumerate(inputs):
                # color codes
                # * green if prediction is correct and adv fails
                # * red if prediction is wrong and adv doesn't flip
                # * blue if prediction is correct and adv flips
                # * black if prediction is wrong and adv doesn't flip
                adv_swp = torch.argmax(init_outputs[evt]) == \
                    torch.argmax(perturbed_outputs[evt])
                init_pred_correct = torch.argmax(init_outputs[evt]) == \
                    labels[evt]
                color = 'k'
                if init_pred_correct and adv_swp:
                    color = 'g'
                if init_pred_correct and (not adv_swp):
                    color = 'b'
                if (not init_pred_correct) and (not adv_swp):
                    color = 'r'
                titlestr = data_manager.label_names[labels[evt]] + '\n' + \
                    str(sm(init_outputs[evt]).numpy()) + ' atk-> ' + \
                    str(sm(perturbed_outputs[evt]).numpy())
                ax = plt.subplot(gs[evt])
                ax.axis('on')
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
                _ = ax.imshow(
                    np.moveaxis(img_tnsr.numpy().astype(dtype), 0, -1))
                plt.title(titlestr, fontsize=6, color=color)
                # plt.tight_layout()
            pdf.savefig()


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
