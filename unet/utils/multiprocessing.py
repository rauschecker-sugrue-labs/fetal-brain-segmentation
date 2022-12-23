from tqdm.auto import tqdm
from multiprocessing import Pool, get_context
from functools import partial


def run_multiprocessing(func, iterable_arguments, fixed_arguments={}, num_processes=12, title="", position=0):
    """ Performs the task indicated in `func` in parallel on `num_processes` CPUs
    Arguments:
        func: function to be executed in parallel
        iterable_arguments: list of inputs to be served to func on different processes
        fixed_arguments: dictionary containing other arguments necessary for func
        num_processes: number of CPU to use
        title: string to name the progress bar
        position: position of tqdm bar. If set to -1, no progress bar.
    Returns:
        the results of func as a list
    Example:
        run_multiprocessing(binarize, images, {'prob_seg_dir':input_dir, 'bin_seg_dir':output_dir, 'threshold':0.6})
    """
    func = partial(func, **fixed_arguments)
    leave_bar = position==0  
    result_list_tqdm = []
    if num_processes > 1:
        with get_context('fork').Pool(processes=num_processes) as pool:
            if position==-1:
                for result in pool.imap(func=func, iterable=iterable_arguments):
                    result_list_tqdm.append(result)
            else:
                for result in tqdm(pool.imap(func=func, iterable=iterable_arguments), total=len(iterable_arguments), desc=title, position=position, leave=leave_bar):
                    result_list_tqdm.append(result)
            return result_list_tqdm
    else:
        for item in tqdm(iterable_arguments, desc=title, position=position, leave=leave_bar):
            result_list_tqdm.append(func(item))
        return result_list_tqdm