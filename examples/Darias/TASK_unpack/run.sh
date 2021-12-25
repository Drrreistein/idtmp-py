#!/bin/bash
# for mlp i dir 4 used model: ../training_cnn_simple/mlp_tf_dir4_21801_56.model/
# for mlp i all dir used model: ../training_cnn_simple/mlp_tf_dirall_21801_40.model/
# for cnn i used model: ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/

python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.1 >> log_fc/mlp_statistic_01.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.3 >> log_fc/mlp_statistic_03.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.5 >> log_fc/mlp_statistic_05.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.7 >> log_fc/mlp_statistic_07.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.9 >> log_fc/mlp_statistic_09.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.2 >> log_fc/mlp_statistic_02.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.4 >> log_fc/mlp_statistic_04.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.6 >> log_fc/mlp_statistic_06.log
python3 run_idtmp_unpack.py -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -n 50 -i 200 -t 0.8 >> log_fc/mlp_statistic_08.log

python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.1 >> log_fc/cnn_statistic_01.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.3 >> log_fc/cnn_statistic_03.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.5 >> log_fc/cnn_statistic_05.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.7 >> log_fc/cnn_statistic_07.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.9 >> log_fc/cnn_statistic_09.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.2 >> log_fc/cnn_statistic_02.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.4 >> log_fc/cnn_statistic_04.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.6 >> log_fc/cnn_statistic_06.log
python3 run_idtmp_unpack.py -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -n 50 -i 200 -t 0.8 >> log_fc/cnn_statistic_08.log


# python3 run_idtmp_unpack.py -r 0.10 -n 20 -c 3 -f ../training_cnn_simple/mlp_tf_dir4_21801_56.model/ >> log_fc/idtmp_010_mlp_new.log
# cd log_fc
# ./log_2_txt.sh idtmp_010_mlp_new
# cd ..

# python3 run_idtmp_unpack.py -r 0.08 -n 20 -c 3 -f ../training_cnn_simple/mlp_tf_dir4_21801_56.model/ >> log_fc/idtmp_008_mlp_new.log
# cd log_fc
# ./log_2_txt.sh idtmp_008_mlp_new
# cd ..

# python3 run_idtmp_unpack.py -r 0.06 -n 20 -c 3 -f ../training_cnn_simple/mlp_tf_dir4_21801_56.model/ >> log_fc/idtmp_006_mlp_new.log
# cd log_fc
# ./log_2_txt.sh idtmp_006_mlp_new
# cd ..

# python3 run_idtmp_unpack.py -r 0.04 -n 20 -c 3 -f ../training_cnn_simple/mlp_tf_dir4_21801_56.model/ >> log_fc/idtmp_004_mlp_new.log
# cd log_fc
# ./log_2_txt.sh idtmp_004_mlp_new
# cd ..

# python3 run_idtmp_unpack.py -r 0.02 -n 20 -c 3 -f ../training_cnn_simple/mlp_tf_dir4_21801_56.model/ >> log_fc/idtmp_002_mlp_new.log
# cd log_fc
# ./log_2_txt.sh idtmp_002_mlp_new
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