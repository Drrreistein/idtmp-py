#!/bin/bash

python3 run_idtmp_unpack.py 0 0.1 20 20 1 test >> log3/idtmp_010_fc.log
python3 run_idtmp_unpack.py 0 0.08 20 20 1 test >> log3/idtmp_008_fc.log
python3 run_idtmp_unpack.py 0 0.06 20 20 1 test >> log3/idtmp_006_fc.log
python3 run_idtmp_unpack.py 0 0.04 20 20 1 test >> log3/idtmp_004_fc.log
python3 run_idtmp_unpack.py 0 0.02 20 20 1 test >> log3/idtmp_002_fc.log