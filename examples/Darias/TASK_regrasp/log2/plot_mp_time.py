

import os, sys
import numpy as np
from IPython import embed
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


filenames = [ 'idtmp_regrasp_010_500.log', 'idtmp_regrasp_005_500.log']
# filenames = [ 'idtmp_regrasp_010_fc_500.log', 'idtmp_regrasp_010_500.log', 'idtmp_regrasp_005_500.log', 'idtmp_regrasp_005_fc_500.log']
data_pd = defaultdict(list)
# data_pd_010 = defaultdict(list)

for filename in filenames:
    if os.path.exists('tmp.txt'):
        os.remove('tmp.txt')
    os.system(f'grep -i "motion refine failed\|task and motion plan found\|current task plan is infeasible\|current task plan is feasible" {filename} >> tmp.txt')

    with open('tmp.txt', 'r') as f:
        mp_times = []
        fc_times = []

        mp_num = 0
        fc_num = 0
        pointer = 0
        for line in f:
            # print(line)
            if 'current task plan is infeasible' in line:
                fc_num += 1
            if 'fc' in filename and 'current task plan is feasible' in line:
                mp_num += 1
            if 'fc' not in filename and 'motion refine failed' in line:
                mp_num += 1
            if 'task and motion plan found' in line:
                if mp_num==0:
                    continue
                if '005' in filename:
                    data_pd['mp_times'].append(mp_num)
                    data_pd['resolution'].append('idtmp_005')

                if 'fc' in filename:
                    data_pd['feasible_check'].append('yes')
                else:
                    data_pd['feasible_check'].append('no')

                if '010' in filename:
                    data_pd['mp_times'].append(mp_num)
                    data_pd['resolution'].append('idtmp_010')

                mp_times.append(mp_num)
                fc_times.append(fc_num+mp_num)
                mp_num = 0
                fc_num = 0

data_pd = pd.DataFrame(data_pd)

medianprops = dict(markerfacecolor='r', color='r')
max_y = max(data_pd['mp_times']) * 1.1

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(5, 6))
# ax.set_title("total planning time", fontdict=dict(fontsize=20))
# Plot the orbital period with horizontal boxes
# hue='feasible_check',
sns.boxplot(x="resolution", y='mp_times', data=data_pd, orient="v", 
    linewidth=1, medianprops=medianprops, whis=5, width=0.5,
    fliersize=0, color=[1,1,1])

ax.set_ylabel("motion planner calling", fontdict=dict(fontsize=14))
ax.set_xlabel("", fontdict=dict(fontsize=14))

ax.set_ylim([0,max_y])
xlabels = ['idtmp_010','idtmp_005']
ax.set_xticklabels(xlabels, fontdict=dict(fontsize=14))
ax.tick_params(labelrotation=30)
plt.tight_layout()
# sns.despine(trim=True, left=True)
# plt.savefig("total_planning_time_idtmp_fc.pdf")
plt.show()

