from run_idtmp_random import *



def test():
    file_list = os.listdir('random_scenes')

    for _ in range(100):
        file = np.random.choice(file_list)
        cmds = [f"python3 run_idtmp_random.py -i 100 -n 5 -c 4 -l {file} >> log/idtmp_statistic_timing.log", 
                f"python3 run_idtmp_random.py -i 100 -n 5 -c 3 -f ../training_cnn_simple/mlp_tf_dirall_21801_40.model/ -l {file} >> log/idtmp_mlp_statistic_timing.log",
                f"python3 run_idtmp_random.py -i 100 -n 5 -c 2 -f ../training_cnn_simple/cnn_fv_100_100_60529_dir4_30_freeze_conv.model/ -l {file} >> log/idtmp_cnn_statistic_timing.log"]

        procs = []
        for i in range(3):
            proc = Process(target=os.system, args=(cmds[i],))
            proc.start()
            procs.append(proc)
        
        for i in range(3):
            procs[i].join()


if __name__=='__main__':
    test()