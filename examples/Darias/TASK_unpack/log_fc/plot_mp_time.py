

import os, sys
import numpy as np
from IPython import embed
from collections import defaultdict
import pandas as pd

filenames = ["idtmp_010.log",
"idtmp_008.log",
"idtmp_006.log",
"idtmp_004.log",
"idtmp_002.log"]
data_pd = defaultdict(list)
# data_pd_010 = defaultdict(list)

for filename in filenames:
    if os.path.exists('tmp.txt'):
        os.remove('tmp.txt')
    os.system(f'grep -i "failed operator\|final_visits" {filename} >> tmp.txt')

    with open('tmp.txt', 'r') as f:
        mp_times = []
        fc_times = []

        mp_num = 0
        fc_num = 0
        pointer = 0
        for line in f:
            # print(line)

            if 'failed operator' in line:
                mp_num += 1
            if 'final_visits' in line:
                data_pd['mp_times'].append(mp_num)
                data_pd['resolution'].append(filename[:-4])

                mp_times.append(mp_num)
                mp_num = 0

data_pd = pd.DataFrame(data_pd)
