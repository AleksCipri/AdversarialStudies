#!/bin/bash

FGSMSTR="0_010"
FGSMSTR="0_000"
FGSMSTR="0_050"

EXE="dset_viz_attacked_stargalaxy.py"
ARGS="--batch-size 4"
ARGS+=" --data-full-path ./fgsm_${FGSMSTR}.hdf5"
ARGS+=" --num-batches 10"
ARGS+=" --pdf-name evts_${FGSMSTR}.pdf"

# show exe and args
cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
