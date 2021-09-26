#!/bin/bash
name=$1
reso=$2
for i in {1..90}; do
(   
    if [ `ls $name* | wc -l` = 0 ]; then
        filename=$name.log
    else
        filename=$name`ls $name* | wc -l`.log
    fi

    python3 run_idtmp.py 0 $reso >> "log/$filename"
)
done
