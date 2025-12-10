#!/bin/bash
#cal_v.sh

for p in */*.prmtop; do
    PRMTOP="${p%/}"
    echo $PRMTOP
    python calculate_volume.py $PRMTOP ${PRMTOP:0:-10}.bak/109345/mdcrd_v --output ${PRMTOP:0:-10}_v.csv
done

