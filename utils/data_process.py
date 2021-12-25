from textwrap import indent
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import repeat
import seaborn as sns
import pandas as pd
from collections import defaultdict
from IPython import embed
import os, sys

def duration_vs_resolution():
    # times = {'t010':np.loadtxt('total_time_010.txt'), 
    #         't008':np.loadtxt('total_time_008.txt'), 
    #         't006':np.loadtxt('total_time_006.txt'), 
    #         't004':np.loadtxt('total_time_004.txt'),
    #         't002':np.loadtxt('total_time_002.txt')}

    # times_mat = np.c_[times['t010'][:50], times['t008'],times['t006'],times['t004'],times['t002']]
    # labels = ['0.10','0.08','0.06','0.04','0.02']
    # ## by matplotlib
    # plt.boxplot(times_mat, whis=(0,100), vert=False,  labels=labels)
    # plt.ylabel('resolution')
    # plt.xlabel('total planning time')
    # plt.show()
    
    ##  by seaborn 
    times_dataset = {}
    times_dataset['name']=[]
    times_dataset['time']=[]
    for s in ['010','008','006','004','002']:
        t_array = np.loadtxt(f'total_time_{s}.txt')
        for i,t in enumerate(t_array):
            times_dataset['name'].append(f'{s[0]}.{s[1:]}')
            times_dataset['time'].append(t)
    times_dataset = pd.DataFrame(times_dataset)
    embed()
    sns.set_theme(style="ticks")
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    # ax.set_xscale("log")
    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="name", y="time", data=times_dataset)

    # Add in points to show each observation
    sns.stripplot(x="name", y="time", data=times_dataset,
              size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="total planning time / s", xlabel="discrete sampling resolution / m")
    sns.despine(trim=True, left=True)
    plt.show()

def auto_convert_num(s:str):
    if '.' in s:
        return float(s)
    else:
        return int(s)

def data_process():
    roh_data=dict()
    # for file in ['010','008','006','004', '002']:
    #     data_tmp = defaultdict(list)
    #     file_name = f'time{file}.log'
    #     with open(file_name, 'r') as f:
    #         for s in f:
    #             s_list = s.split()
    #             data_tmp[s_list[0]].append(auto_convert_num(s_list[1]))
    #     roh_data[file] = data_tmp
    # embed()
    dataset = defaultdict(list)
    for file in ['etamp', '010','008','006','004', '002']:
        if file=='etamp':
            file_name = 'etamp_unpack_clean.log'
        else:
            file_name = f'time{file}.log'
        with open(file_name, 'r') as f:
            num = 0
            for s in f:
                s_list=s.split()
                if s_list[0]=='motion_refiner_time':
                    num += 1
                    dataset['motion_refiner_time'].append(auto_convert_num(s_list[1]))
                elif s_list[0]=='task_plan_time':
                    dataset['task_plan_time'].append(auto_convert_num(s_list[1]))
                elif s_list[0]=='total_planning_time':
                    dataset['total_planning_time'].append(auto_convert_num(s_list[1]))
                elif s_list[0]=='task_plan_counter':
                    dataset['task_plan_counter'].append(auto_convert_num(s_list[1]))
            dataset['resolution'].extend(list(np.repeat(file, num)))

    dataset['total_div_task'] = list(np.array(dataset['total_planning_time']) / np.array(dataset['task_plan_counter']))
    for k, v in dataset.items():
        print(k, len(v))

    data_pd = pd.DataFrame(dataset)

    return data_pd

def txt_to_pandas(file_list):
    dataset = defaultdict(list)
    reso = []
    for file_name in file_list:
        with open(file_name, 'r') as f:
            num = 0
            for s in f:
                s_list=s.split()
                dataset[s_list[0]].append(auto_convert_num(s_list[1]))
                num += 1
            num /= len(dataset.keys())
            reso.extend(list(np.repeat(file_name[:-4], int(num))))

    dataset['resolution'] = reso
    try:
        dataset['time_per_task'] = list(np.array(dataset['total_planning_time']) / np.array(dataset['final_visits']))
    except:
        print(f"no specified item in dataset")
        
    for k, v in dataset.items():
        print(k, len(v))

    data_pd = pd.DataFrame(dataset)
    return data_pd

def txt_to_dict(file_list):
    dataset = defaultdict(list)
    for file_name in file_list:
        with open(file_name, 'r') as f:
            for s in f:
                s_list=s.split()
                dataset[s_list[0]].append(auto_convert_num(s_list[1]))
        
    for k, v in dataset.items():
        print(k, len(v))

    return dataset

def box_plot(data_pd):

    indices = list(data_pd.axes[1])
    indices.remove('resolution')
    medianprops = dict(markerfacecolor='r', color='r')

    for ind in indices:
        # sns.set_theme(style="ticks")
        # Initialize the figure with a logarithmic x axis
        f, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f'{ind}')
        # ax.set_xscale("log")
        # Plot the orbital period with horizontal boxes
        sns.boxplot(y=ind, x="resolution", data=data_pd, orient="v", 
            linewidth=1, medianprops=medianprops, whis=5, width=0.5,
            fliersize=0, color=[1,1,1])

        # Add in points to show each observation
        # sns.stripplot(x="resolution", y=ind, data=data_pd,
        #         size=4, color=".3", linewidth=0)
        # Tweak the visual presentation
        # ax.xaxis.grid(True)

        # ax.set(ylabel=f"{ind} / s", xlabel="discrete sampling resolution / m")
        ax.set(ylabel=f"{ind}", xlabel='')
        ax.set_xticklabels(['idtmp_010','idtmp_008','idtmp_006','idtmp_004','idtmp_002'], fontdict=dict(fontsize=10))
        # sns.despine(trim=True, left=True)
        plt.tight_layout()
        plt.savefig(f"{ind}.pdf")
        plt.show()

def box_plot_fc_vs_none(filelist_no, filelist_fc):

    pd_no = txt_to_pandas(filelist_no)
    pd_fc = txt_to_pandas(filelist_fc)
    pd_no['feasible_check'] = len(pd_no) * ['no']
    pd_fc['feasible_check'] = len(pd_fc) * ['yes']
    tmp=[]
    for name in pd_fc['resolution']:
        tmp.append(name[:-3])
    pd_fc['resolution'] = tmp
    data_pd = pd.concat([pd_fc, pd_no])


    medianprops = dict(markerfacecolor='r', color='r')
    max_y = max(data_pd['total_planning_time']) * 1.1

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(5, 6))
    # ax.set_title("total planning time", fontdict=dict(fontsize=20))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="resolution", y='total_planning_time', data=data_pd, orient="v", hue='feasible_check',
        linewidth=1, medianprops=medianprops, whis=5, width=0.5,fliersize=0, color=[1,1,1])

    # Add in points to show each observation
    # sns.stripplot(x="resolution", y=ind, data=data_pd,
    #         size=4, color=".3", linewidth=0)
    # Tweak the visual presentation
    # ax.yaxis.grid(True)

    ax.set_ylabel("total planning time [s]", fontdict=dict(fontsize=14))
    ax.set_xlabel("", fontdict=dict(fontsize=14))
    
    ax.set_ylim([0,max_y])
    xlabels = ['idtmp_010','idtmp_008','idtmp_006','idtmp_004','idtmp_002']
    ax.set_xticklabels(xlabels, fontdict=dict(fontsize=14))
    ax.tick_params(labelrotation=30)
    plt.tight_layout()
    plt.show()

def get_table_data_of_all():
    
    file_mlp = ['idtmp_010_mlp.txt', 'idtmp_008_mlp.txt', 'idtmp_006_mlp.txt', 'idtmp_004_mlp.txt', 'idtmp_002_mlp.txt']
    file_no = ['idtmp_010.txt', 'idtmp_008.txt', 'idtmp_006.txt', 'idtmp_004.txt', 'idtmp_002.txt', 'etamp_con.txt']
    file_cnn = ['idtmp_010_cnn_new.txt', 'idtmp_008_cnn_new.txt', 'idtmp_006_cnn_new.txt', 'idtmp_004_cnn_new.txt', 'idtmp_002_cnn_new.txt']

    data_pd = txt_to_pandas(file_no)
    for key in list(data_pd.keys()):
        print(key)
        for s in np.unique(data_pd['resolution']):
            if key=='resolution':
                continue
            tmp=np.array(data_pd[key][data_pd["resolution"]==s])
            print(s, np.mean(tmp))


    pass

if __name__=='__main__':
    embed()
    # duration_vs_resolution()

    dataset = data_process()

    box_plot(dataset)