#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/Workspace"
EXE="vanilla_stargalaxy.py"
DAT=`date +%s`

ARGS="--batch-size 32"
ARGS+=" --ckpt-path sg_ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --git-hash `git describe --abbrev=12 --dirty --always`"
ARGS+=" --log-freq 20"
ARGS+=" --log-level INFO"
ARGS+=" --num-epochs 1"
# ARGS+=" --short-test"
ARGS+=" --show-progress"
ARGS+=" --test"
ARGS+=" --tnsrbrd-out-dir /tmp/stargalaxy/tnsrbrd${DAT}"
ARGS+=" --train"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
