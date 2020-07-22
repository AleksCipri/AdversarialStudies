# Top-level Contents

* `attack_stargalaxy.py` -- script to run an attacker on the star-galaxy
  dataset.
* `check_coverage.sh` -- run `coverage` over test suite (via `run_tests.py`)
  and report unit test coverage.
* `cleanup.sh` -- remove development detritus, pyc files, etc.
* `dset_viz_attacked_stargalaxy.py` -- visualize star-galaxy data attacked by
  an adversary.
* `dset_viz_stargalaxy.py` -- visualize star-galaxy data.
* `notebooks` -- Jupyter notebooks for using the libraries here.
  See below for more information.
* `ptlib/` -- lib dir for PyTorch code modules.
* `requirements.txt` -- packages needed to run everything.
* `run_attack_stargalaxy.sh` -- bash script to wrap `attack_stargalaxy.py` with
  convenience arguments.
* `run_dset_viz_atacked_stargalaxy.sh` -- bash script to wrap
  `dset_viz_attacked_stargalaxy.py` with convenience arguments.
* `run_tests.py` -- run all tests in `tests/`; generally best to avoid running
  this directly and instead run `check_coverage.sh`.
* `run_update_todos.sh` -- search code for "TODO" marks and drop them into the
  `TODO.md` file.
* `run_vanilla_fashion.sh` -- wrapper script for Fashion MNIST classifier.
* `run_vanilla_stargalaxy.sh` -- wrapper script for the star-galaxy classifier.
* `tests/` -- lib dir for `unittest` modules.
* `TODO.md` -- programatically generated file with list of TODOs.
* `vanilla_fashion.py` -- script to run a Fashion MNIST classifier.
* `vanilla_stargalaxy.py` -- script to run a star-galaxy classifier.

## Data

Tensors are stored using PyTorch convnetions - [Number, Channel, Width, Height]

### Star-Galaxy

Images courtesy of Joao Caldeira (provenance unknown to me, but likely known to
him). HDF5 data is available here:

https://github.com/gnperdue/RandomData/tree/master/StarGalaxy

e.g.

```
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/StarGalaxy/stargalaxy_real_pt_test.hdf5
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/StarGalaxy/stargalaxy_real_pt_train.hdf5
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/StarGalaxy/stargalaxy_real_pt_valid.hdf5
```

**NOTE** -- the images stored in that Git repo are stored as `uint8`, but the
code expects floating point numbers. There is a script in the
`RandomData/StarGalaxy` package on
[GitHub](https://github.com/gnperdue/RandomData/tree/master/StarGalaxy) for
doing the conversion using the
[hdf5_manipulator](https://github.com/gnperdue/hdf5_manipulator) package of
utilities. (We are not storing the floats version in GitHub because those files
are too large -- even the `uint8` files are running up against their
guidelines.)

### Fashion MNIST

HDF5 data is available here:

https://github.com/gnperdue/RandomData/tree/master/hdf5

e.g.

```
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/hdf5/fashion_test.hdf5
wget https://raw.githubusercontent.com/gnperdue/RandomData/master/hdf5/fashion_train.hdf5
```

(Use 'test' for 'validation' -- no real 'test' file offered for this dset
because it is _fashion MNIST_ for crying out loud...)

## Notebooks

The following notebooks provide a demonstration of the API. In order, they are:

1. `star_galaxy.ipynb` -- train a simple "star-galaxy" classifier and store
the trained model locally.
2. `attack_star_galaxy_fgsm.ipynb` -- run the Fast Gradient Sign Method attack
on the star-galaxy dataset for handful of attack strengths, using the model
trained in the previous notebook, and save the attacked images in local
HDF5 files.
3. `visualiz_fgsm_attack_star_galaxy.ipynb` -- visualize the "attacked" images
for a given attack strength and save copies to a PDF for later viewing.
