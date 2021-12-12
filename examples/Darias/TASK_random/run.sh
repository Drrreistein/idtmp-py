#!/bin/bash

python3 run_idtmp_unpack.py 0 0.10 60 20 2 unpack_010_cnn >> log_fc/idtmp_010_cnn.log
cd log_fc
./log_2_txt.sh idtmp_010_cnn
cd ..

python3 run_idtmp_unpack.py 0 0.08 80 20 2 unpack_008_cnn >> log_fc/idtmp_008_cnn.log
cd log_fc
./log_2_txt.sh idtmp_008_cnn
cd ..

python3 run_idtmp_unpack.py 0 0.06 60 20 2 unpack_006_cnn >> log_fc/idtmp_006_cnn.log
cd log_fc
./log_2_txt.sh idtmp_006_cnn
cd ..

python3 run_idtmp_unpack.py 0 0.04 40 20 2 unpack_004_cnn >> log_fc/idtmp_004_cnn.log
cd log_fc
./log_2_txt.sh idtmp_004_cnn
cd ..

python3 run_idtmp_unpack.py 0 0.02 20 20 2 unpack_002_cnn >> log_fc/idtmp_002_cnn.log
cd log_fc
./log_2_txt.sh idtmp_002_cnn
cd ..

# python3 run_idtmp_unpack.py 0 0.1 20 20 1 test >> log3/idtmp_010_fc.log
# python3 run_idtmp_unpack.py 0 0.08 20 20 1 test >> log3/idtmp_008_fc.log
# python3 run_idtmp_unpack.py 0 0.06 20 20 1 test >> log3/idtmp_006_fc.log
# python3 run_idtmp_unpack.py 0 0.04 20 20 1 test >> log3/idtmp_004_fc.log
# python3 run_idtmp_unpack.py 0 0.02 20 20 1 test >> log3/idtmp_002_fc.log