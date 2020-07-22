#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/Workspace"
EXE="attack_stargalaxy.py"
DAT=`date +%s`

ARGS="--ckpt-path sg_ckpt.tar"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --epsilons 0.,0.01,0.05"
ARGS+=" --git-hash `git describe --abbrev=12 --dirty --always`"
ARGS+=" --log-freq 5"
ARGS+=" --log-level INFO"
# ARGS+=" --log-level INFO"
ARGS+=" --short-test"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
