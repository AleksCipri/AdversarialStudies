#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/Workspace"
EXE="diffevo_attack_stargalaxy.py"
DAT=`date +%s`

ARGS="--ckpt-path sg_ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --git-hash `git describe --abbrev=12 --dirty --always`"
ARGS+=" --log-freq 5"
ARGS+=" --log-level INFO"
ARGS+=" --max-examps 1"
ARGS+=" --max-iterations 100"
ARGS+=" --num-pixels 1"
ARGS+=" --pop-size 400"
ARGS+=" --short-test"
ARGS+=" --targeted"
ARGS+=" --verbose"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
