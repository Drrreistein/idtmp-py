import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython import embed
import pickle as pk



def main():
    sum_cnn = defaultdict(list)
    for i in range(1,10):
        file_name = f'cnn_statistic_0{i}.txt'
        tmp = dict()
        with open(file_name, 'r') as f:
            for s in f:
                s_list = s[:-1].split(' ')
                tmp[s_list[0]] = int(s_list[-1])
        for k,v in tmp.items():
            sum_cnn[k].append(v)

    sum_mlp = defaultdict(list)
    for i in range(1,10):
        file_name = f'mlp_statistic_0{i}.txt'
        tmp = dict()
        with open(file_name, 'r') as f:
            for s in f:
                s_list = s[:-1].split(' ')
                tmp[s_list[0]] = int(s_list[-1])
        for k,v in tmp.items():
            sum_mlp[k].append(v)
        

    res = dict()
    for k,v in sum_cnn.items():
        val = np.round(np.array(v)/np.sum(np.array(list(sum_cnn.values())), axis=0), 5)
        res[k+'_cnn'] = val

    for k,v in sum_mlp.items():
        val = np.round(np.array(v)/np.sum(np.array(list(sum_mlp.values())), axis=0), 5)
        res[k+'_mlp'] = val
    dat_pd = pd.DataFrame(res)
    return dat_pd