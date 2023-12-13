import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy
import pandas
import argparse
from data.steersim import steersimProcess
from data.agent_grouping import AgentGrouping

def load_data(file_path):
    arr, _  = steersimProcess.read_trajectory_binary(file_path)
    d = pandas.DataFrame(arr[:, [0, 1, 13, 15]], columns='nframe aid x z'.split())
    d['nframe'] = d.nframe.astype(int)
    d['aid'] = d.aid.astype(int)

    clamp_max = d[d.z < 1].groupby('aid').nframe.max()
    merged = pandas.merge(d, clamp_max, on="aid", suffixes=("", "_clamp"))
    filtered = merged[merged.nframe > merged.nframe_clamp].copy()
    filtered['nframe'] = filtered.nframe - filtered.nframe_clamp
    return filtered.drop('nframe_clamp', axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str)
    parser.add_argument('file2', type=str)
    args = parser.parse_args()

    df1 = load_data(args.file1)
    df2 = load_data(args.file2)

    df2.aid = df2.aid + 1000000
    df = df1.append(df2)

    print('Computing MAE...')
    mae_list = []
    for i in df1.aid.unique():
        mae = AgentGrouping.compute_MAE(df, i, i+1000000)
        mae_list.append(mae)

    print('MAE: ', numpy.mean(mae_list))
