from collections import defaultdict
from unittest import result
import pandas as pd
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import os


def get_results(folders, data, feature, metric):
    results = defaultdict(list)
    for folder in folders:
        for ack in glob.glob(folder + '/' + data + '/' + feature + '/*'):
            print('ack',ack)
            _ack = ack.replace(data + '/' + feature + '/', '')
            if os.name == 'nt': # if on windows os
                _ack =_ack=_ack.split('\\')[-1] 
            results[_ack] = defaultdict(list)
            for seed in glob.glob(ack + '/*'):
                _seed = seed.replace(ack + '/', '')
                if os.name == 'nt': # if on windows os
                    _seed = _seed.split('\\')[-1] 
                #print(_ack, _seed)
                d = pd.read_pickle(seed + '/results.pickle')
                for cycle in range(len(d)):
                    results[_ack][_seed].append(d[cycle][metric])
    return results


def explode_cycles(df):
    return df.explode(df.columns.values.tolist()).reset_index(drop=True)


def create_df_per_acq_function(df):
    dfs = []
    for acq in df.index:
        acq_df = df[df.index==acq]
        acq_df = explode_cycles(acq_df)
        dfs.append(acq_df)
    return dfs


def plot_acq_together(df,acq_name,color,data,feat,metric):
    D=df.to_numpy().astype(float)
    sems = np.std(D, axis=1, ddof=1) / np.sqrt(np.size(D))
    means =np.mean(D, axis = 1)

    plt.xlabel('cycles')
    plt.ylabel(f'metric ({metric})')
    plt.title(f'Mean and stderr over cycles for all acquisition functions\nfor {data} and {feat}')

    plt.plot([0,1,2,3,4,5,6,7],
                        means,
                        lw=1.5,
                        ls="-", label=acq_name, color=color,
                    )

    plt.fill_between(x=[0,1,2,3,4,5,6,7],
                        y1=means - 1.96 * sems,
                        y2=means + 1.96 * sems,
                        alpha=0.3, color=color
                    )
    plt.legend(loc='best')
    plt.savefig(f'all_aqc_({data},{feat},{metric}).jpg',dpi=1000)


def run(folders, dataset,feat,metric):
    colors=['red','greenyellow','violet','cyan','yellow','blue','magenta','lime','gold','orangered','steelblue']
    results = get_results(folders, dataset, feat, metric)
    df = pd.DataFrame.from_dict(results, orient='index')
    if len(colors)>=len(df):
        dfs = create_df_per_acq_function(df)
        for i in range(len(dfs)):
            plot_acq_together(dfs[i], df.index[i], colors[i], dataset, feat, metric)
            if i == len(dfs) - 1:
                plt.show()
                plt.clf()
    else:
        print(f'ERROR: add more colors to match the number of acquisition functions; len(colors): {len(colors)} < len(acq functions): {len(dfs)}')
    


data = sys.argv[1]
feature = sys.argv[2]
metric = sys.argv[3]

folders = ['output','output_rnd']

run(data,feature,metric)


import pdb; pdb.set_trace()
