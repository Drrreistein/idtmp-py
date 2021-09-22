#!/bin/bash
name=$1
for i in {1..10}; do
(
    # filename = $name`ls $name* | wc -l`.log
    python3 run_idtmp.py >> "log/$name"
)
done