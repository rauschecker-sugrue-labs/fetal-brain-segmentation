# To measure timing
#' http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(note='', print_to_screen=True, restart=False):
    import time
    if 'startTime_for_tictoc' in globals():
        elapsed_time = (time.time() - startTime_for_tictoc)
        if print_to_screen: print(f"{note}. Elapsed time is {elapsed_time:.3f} seconds.")
        if restart: tic()
        return elapsed_time
    else:
        print("Toc: start time not set")
        return
