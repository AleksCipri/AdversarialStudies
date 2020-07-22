#!/bin/bash

DATADIR="/Users/perdue/Dropbox/Data/Workspace"
FGSMSTR="0_000"
FGSMSTR="0_010"
FGSMSTR="0_050"

EXE="dset_viz_stargalaxy.py"
ARGS="--batch-size 5"
ARGS+=" --data-dir ${DATADIR}"
ARGS+=" --file-override ./fgsm_${FGSMSTR}.hdf5"
ARGS+=" --num-batches 8"
ARGS+=" --pdf-name evts_${FGSMSTR}.pdf"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
