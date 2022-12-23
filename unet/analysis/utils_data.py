import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

def missing_columns(li1, li2, name=None):
    """ prints the names of missing data points from one list to the other. name is a label to help if this function is called several times.
    """
    l1 = len(li1)
    l2 = len(li2)
    if l2 < l1:
        if name:
            print(f'Missing {l1-l2} columns in {name}!\t{diff_2lists(li1, li2)}')
        else:
            print(f'Missing {l1-l2} columns!\t{diff_2lists(li1, li2)}')


def diff_2lists(li1, li2):
    return list(list(set(li2)-set(li1)) + list(set(li1)-set(li2)))

def exp_df(csv_path, exp_name, sep=' '):
    """ Return DataFrame with exp_name suffixed to columns and 'Patient' as index
    """
    df = pd.read_csv(csv_path)
    # df = df[['Patient', 'd']]
    df.set_index('Patient', inplace=True)
    return df.add_prefix(exp_name + sep)

def filter_metric(df, metric):
    """ Return DataFrame with only the columns with appropriate metric
    """
    col_metric = [col for col in df.columns if col.endswith(metric)]
    df_filter = df[col_metric]
    return remove_metric_colnames(df_filter, metric)

def remove_metric_colnames(df, metric):
    new_cols = {}
    for col in df.columns:
        if col.endswith(metric):
            new_cols[col] = col.split(metric)[0][:-1]
    return df.rename(columns=new_cols)