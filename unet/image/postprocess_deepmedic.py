#%% Script used to post process CNN output patch for deepmedic.

import numpy as np
import SimpleITK as sitk
from utils.multiprocessing import run_multiprocessing
from multiprocessing import sharedctypes

# This function assemble segmentation patches into - test version
# Input:
#   patches: list of n tuple, n being the number of patch this patient has. 
#            Each tuple of the following structure 
#            (ph[0], pl[0], seg, cpts, shape, disease, patient)
#            The only useful information here is just the cpts and shape
#   patches_pred: numpy array of shape [n, hs, ws, ds, C]. hs, ws, ds being size of the segmentation patch, 
#         C being probability of each label
# Output:
#   seg: numpy array of shape [H, W, D, C], H, W, D being size of complete image.
def assemSegFromPatches_dir(shape, cpts, patches_pred, num_processes=1, position=None):
    # Get shape param
    H, W, D = shape
    n, hs, ws, ds, C = patches_pred.shape

    im_shape = np.array([H, W, D])
    half_shape = (np.array([hs, ws, ds])/2).astype(np.int32)
    padded_shape = (im_shape + half_shape * 2).astype(np.int32)

    # Pad the segmentation so that the edge cases are handled
    seg = np.ctypeslib.as_ctypes(np.zeros(list(padded_shape) + [C]))
    rep = np.ctypeslib.as_ctypes(np.zeros(list(padded_shape) + [C]))

    # Make global C type arrays for populating during multiprocessing
    global seg_shared
    global rep_shared
    seg_shared = sharedctypes.RawArray(seg._type_, seg)
    rep_shared = sharedctypes.RawArray(rep._type_, rep)
    #seg_shared = sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(seg.dtype), seg)
    #rep_shared = sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(rep.dtype), rep)
    args = [(cpt, patch_pred) for cpt, patch_pred in zip(list(cpts),list(patches_pred))]
    res = run_multiprocessing(add_patch, args, 
                            title="Assemble patches",
                            num_processes=num_processes,
                            position=position)

    seg = np.ctypeslib.as_array(seg_shared)
    rep = np.ctypeslib.as_array(rep_shared)
    # Crop out edges
    seg = seg[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]
    rep = rep[half_shape[0]:-half_shape[0], half_shape[1]:-half_shape[1], half_shape[2]:-half_shape[2]]

    # Normalized by repetition
    rep[rep==0] = 1e-6
    seg = seg/rep

    return seg

def add_patch(args):
    cpt, pred = args
    hs, ws, ds, C = pred.shape
    seg_tmp = np.ctypeslib.as_array(seg_shared)
    rep_tmp = np.ctypeslib.as_array(rep_shared)
    try:
        seg_tmp[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += pred
        rep_tmp[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :] += 1
    except ValueError:
        print ("Debug")
        print (cpt[0], cpt[0] + hs, cpt[1], cpt[1] + ws, cpt[2], cpt[2] + ds)
        print (pred.shape)
        print (seg_tmp.shape)
        print (seg_tmp[cpt[0]:cpt[0] + hs, cpt[1]:cpt[1] + ws, cpt[2]:cpt[2] + ds, :].shape)
        return None
