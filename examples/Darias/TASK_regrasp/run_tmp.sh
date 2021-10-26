#!/bin/bash
python3 run_idtmp_regrasp.py 0 0.1 1 500 >> log2/idtmp_regrasp_010.log
python3 runExp_eTAMP_regrasp.py 0 0 1 500 >> log2/etamp_regrasp_con.log
python3 runExp_eTAMP_regrasp.py 0 1 1 500 >> log2/etamp_regrasp_010.log
    