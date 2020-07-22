#!/bin/bash

DAT=`date +%s`
mkdir -p archive

# clean up synth data from testing
rm -fv fash_synth_*.npy
rm -fv fash_synth_*.h5
rm -fv sg_synth_*.npy
rm -fv sg_synth_*.h5
rm -fv atk_sg_synth.h5

# save meaningful logs
mv -fv log_vanilla*.txt archive/
mv -fv log_attack*.txt archive/

# remove testing logs
rm -fv log_run_tests*.txt

# archive training artifacts
FILELIST="sg_ckpt.tar short_test.tar"
for file in $FILELIST
do
  if [[ -e $file ]]; then
    mv -v $file archive/t${DAT}_${file}
  fi
done
rm -fv test_vanilla_trainer.tar

# clean up compiled python
DIRS="ptlib tests"
for dir in $DIRS
do
    rm -rfv ${dir}/__pycache__
done
