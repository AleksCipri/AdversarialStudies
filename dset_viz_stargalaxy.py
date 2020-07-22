'''
plot star-galaxy images
TODO - make a viz library for use in other modules
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ptlib.dataloaders import StarGalaxyDataManager as DataManager

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=6, type=int, help='batch size')
parser.add_argument('--data-dir', default='', type=str, help='data dir')
parser.add_argument('--dtype', default='float64', type=str,
                    help='img data type')
parser.add_argument('--file-override', default=None, type=str,
                    help='overrid for dataset test file full path')
parser.add_argument('--num-batches', default=1, type=int,
                    help='number of batches')
parser.add_argument('--pdf-name', default='evt_all.pdf', type=str,
                    help='output pdf name')


def main(batch_size, data_dir, dtype, file_override, num_batches, pdf_name):
    data_manager = DataManager(data_dir=data_dir)
    if file_override:
        data_manager.testfile = file_override
    print(data_manager.testfile)
    _, _, test_dl = data_manager.get_data_loaders(
        batch_size=batch_size, standardize=False)
    with PdfPages(pdf_name) as pdf:
        for iter_num, (inputs, labels) in enumerate(test_dl, 0):
            if iter_num >= num_batches:
                break
            n_cols = int(np.sqrt(batch_size))
            n_rows = int(np.ceil(batch_size / n_cols))
            # make grid, plots
            gs = plt.GridSpec(n_rows, n_cols)
            for evt, img_tnsr in enumerate(inputs):
                ax = plt.subplot(gs[evt])
                ax.axis('on')
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
                _ = ax.imshow(
                    np.moveaxis(img_tnsr.numpy().astype(dtype), 0, -1))
                plt.title(data_manager.label_names[labels[evt]])
            pdf.savefig()


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
