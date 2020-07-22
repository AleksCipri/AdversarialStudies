#!/bin/bash

DATADIR="${HOME}/Dropbox/Data/RandomData/hdf5"
EXE="vanilla_fashion.py"
DAT=`date +%s`

ARGS="--batch-size 60"
ARGS+=" --ckpt-path ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --log-freq 5"
ARGS+=" --log-level INFO"
ARGS+=" --num-epochs 1"
ARGS+=" --short-test"
ARGS+=" --show-progress"
ARGS+=" --tnsrbrd-out-dir /tmp/fashion/tnsrbrd${DAT}"
# ARGS+=" --tnsrbrd-out-dir /tmp/fashion/tnsrbrd_test"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
