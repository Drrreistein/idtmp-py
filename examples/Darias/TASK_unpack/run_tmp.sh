#!/bin/bash
# python3 run_idtmp_unpack.py 0 0.1 100 20 >> log3/idtmp_unpack_010.log
# python3 run_idtmp_unpack.py 0 0.08 80 20 >> log3/idtmp_unpack_008.log
python3 run_idtmp_unpack.py 0 0.06 60 20 >> log3/idtmp_unpack_006.log
python3 run_idtmp_unpack.py 0 0.04 40 20 >> log3/idtmp_unpack_004.log
python3 run_idtmp_unpack.py 0 0.02 20 20 >> log3/idtmp_unpack_002.log   
spd-say -t female2 "hi lei, simulation done"