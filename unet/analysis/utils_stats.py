from scipy.stats import wilcoxon, ranksums, mannwhitneyu, ttest_ind
import pandas as pd
import numpy as np

def stats(func, base, compare_to, **stats_kwargs):
    """
    Args:
        func: stat function to use
        base: tuple (dataframe, column_names)
        compare_to: dataframe
        **stats_kwargs: additional arguments for stat function (see examples below)
    Examples:
        stats(mannwhitneyu, (penn1, 'col1'), ucsf.columns, **{'alternative':'greater'})
        stats(wilcoxon, (ucsf, 'col2'), ucsf.columns)
        stats(wilcoxon, (ucsf, ['col2', 'col3']), ucsf.columns) # for several analyses
    """
    stats_name = type(func([1],[2])).__name__[:-6]
    sig_label = 'Sig (<5%: ***, <10%: *)'
    list_results = []
    base_df = base[0]
    base_col_names = base[1]
    if type(base_col_names) is str: base_col_names=[base_col_names]
    for base_col_name in base_col_names:
        results = {}
        base_col_name = base_col_name
        base_vals = base_df[base_col_name]
        for col in compare_to.columns:
            if col == base_col_name and base_df is compare_to:
                results[col] = {'Median': np.median(base_vals), f'{stats_name} p-value': 0, sig_label: 'I'}
                continue
            res = func(base_vals, compare_to[col], **stats_kwargs)[1]
            if res <= 0.05:
                sig = '***'
            elif res <= 0.1:
                sig = '*'
            else:
                sig = ''
            results[col] = {'Median': np.median(compare_to[col]), f'{stats_name} p-value': res, sig_label: sig}
        list_results.append(pd.DataFrame(results))

    return pd.concat(list_results)