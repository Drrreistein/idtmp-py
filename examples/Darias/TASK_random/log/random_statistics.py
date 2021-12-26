import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython import embed
import pickle as pk
def read_txt(file):
    data_dic = defaultdict(list)
    with open(file, 'r') as f:
        for s in f:
            s_list = s[:-1].split(' ')
            # print(s_list)
            data_dic[s_list[0]].append(np.array(s_list[1:], dtype=int))
    return data_dic


def main():
    data_cnn = read_txt('cnn_statistic.txt.bak')
    data_mlp = read_txt('mlp_statistic.txt.bak')

    sum_cnn = defaultdict(list)
    sum_mlp = defaultdict(list)

    for k,v in data_cnn.items():
        sum_cnn[k] = list(np.sum(np.array(data_cnn[k]), axis=0))
    for k,v in data_mlp.items():
        sum_mlp[k] = list(np.sum(np.array(data_mlp[k]), axis=0))

    res = dict()
    for k,v in sum_cnn.items():
        val = np.round(np.array(v)/np.sum(np.array(list(sum_cnn.values())), axis=0), 5)
        res[k+'_cnn'] = val

    for k,v in sum_mlp.items():
        val = np.round(np.array(v)/np.sum(np.array(list(sum_mlp.values())), axis=0), 5)
        res[k+'_mlp'] = val
    dat_pd = pd.DataFrame(res)
    return dat_pd