#!/bin/bash

python3 run_idtmp_regrasp.py -r 0.1 -n 20 -c 4 >> log_fc/idtmp_010.log
cd log_fc
./log_2_txt.sh idtmp_010
cd ..

python3 run_idtmp_regrasp.py -r 0.08 -n 20 -c 4 >> log_fc/idtmp_008.log
cd log_fc
./log_2_txt.sh idtmp_008
cd ..

python3 run_idtmp_regrasp.py -r 0.06 -n 20 -c 4 >> log_fc/idtmp_006.log
cd log_fc
./log_2_txt.sh idtmp_006
cd ..

python3 run_idtmp_regrasp.py -r 0.04 -n 20 -c 4 >> log_fc/idtmp_004.log
cd log_fc
./log_2_txt.sh idtmp_004
cd ..

# python3 run_idtmp_regrasp.py -r 0.04 -n 20 -c 4 >> log_fc/idtmp_004.log
# cd log_fc
# ./log_2_txt.sh idtmp_004
# cd ..

# python3 run_idtmp_regrasp.py -r 0.02 -n 20 -c 4 >> log_fc/idtmp_002.log
# cd log_fc
# ./log_2_txt.sh idtmp_002
# cd ..


# python3 run_idtmp_regrasp.py -r 0.1 -n 20 -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ >> log_fc/idtmp_010_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_010_cnn
# cd ..

# python3 run_idtmp_regrasp.py -r 0.08 -n 20 -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ >> log_fc/idtmp_008_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_008_cnn
# cd ..

# python3 run_idtmp_regrasp.py -r 0.06 -n 20 -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ >> log_fc/idtmp_006_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_006_cnn
# cd ..

# python3 run_idtmp_regrasp.py -r 0.04 -n 20 -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ >> log_fc/idtmp_004_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_004_cnn
# cd ..

# python3 run_idtmp_regrasp.py -r 0.02 -n 20 -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ >> log_fc/idtmp_002_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_002_cnn
# cd ..

# python3 run_idtmp_unpack.py 0 0.08 80 20 2 unpack_008_cnn >> log_fc/idtmp_008_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_008_cnn
# cd ..

# python3 run_idtmp_unpack.py 0 0.06 60 20 2 unpack_006_cnn >> log_fc/idtmp_006_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_006_cnn
# cd ..

# python3 run_idtmp_unpack.py 0 0.04 40 20 2 unpack_004_cnn >> log_fc/idtmp_004_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_004_cnn
# cd ..

# python3 run_idtmp_unpack.py 0 0.02 20 20 2 unpack_002_cnn >> log_fc/idtmp_002_cnn.log
# cd log_fc
# ./log_2_txt.sh idtmp_002_cnn
# cd ..

# python3 run_idtmp_unpack.py 0 0.1 20 20 1 test >> log3/idtmp_010_fc.log
# python3 run_idtmp_unpack.py 0 0.08 20 20 1 test >> log3/idtmp_008_fc.log
# python3 run_idtmp_unpack.py 0 0.06 20 20 1 test >> log3/idtmp_006_fc.log
# python3 run_idtmp_unpack.py 0 0.04 20 20 1 test >> log3/idtmp_004_fc.log
# python3 run_idtmp_unpack.py 0 0.02 20 20 1 test >> log3/idtmp_002_fc.log