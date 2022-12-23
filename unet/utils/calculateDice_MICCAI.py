### This script is used to calculate dice
import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
pd.set_option('display.max_rows', None)
import csv
import shutil
from .dice import dice
from utils.multiprocessing import run_multiprocessing
from pathlib import Path
from scipy import ndimage

def compute_volume(sitk_img:sitk.Image) -> float:
    """ Computes volume of binary image
    Arguments:
        sitk_img: sitk binary image or path to it
    Returns:
        float: computed volume
    """
    if isinstance(sitk_img, str) or isinstance(sitk_img, Path):
        sitk_img = sitk.ReadImage(str(sitk_img))
    size_vox = float(sitk_img.GetMetaData('pixdim[1]')) * float(sitk_img.GetMetaData('pixdim[2]')) * float(sitk_img.GetMetaData('pixdim[3]'))
    num_vox = np.sum(sitk.GetArrayFromImage(sitk_img))
    return num_vox*size_vox

def compute_volume_batch(images, num_processes=12):
    """ Compute volumes for a list of binary images
    Arguments:
        images: list(Path)
        num_processes: int. Number of parallel processes to launch
    """
    sitk_images = [sitk.ReadImage(str(image)) for image in images]
    volumes = run_multiprocessing(compute_volume, images, title="Computing volumes", num_processes=num_processes)
    return volumes

def ignore_zeros(a,b,c):
    try:
        result = a / (b + c)
    except ZeroDivisionError:
        result = np.nan
    return result 

def compute_stats(p, gt_dir, pred_dir, inference_only=False):
    lesion_pred = pred_dir / (p + "_binary.nii.gz")
    lesion_gt = gt_dir / (p + "_seg.nii.gz")

    if not os.path.exists(lesion_pred):
        print(f"non-existent: {lesion_pred}")
        return

    # Load prediction and manual segmentation
    pred_sitk = sitk.ReadImage(str(lesion_pred))
    pred = sitk.GetArrayFromImage(pred_sitk)

    # Compute lesions volume
    lesion_volume_pred = compute_volume(pred_sitk)
    if inference_only:
        return (p, lesion_volume_pred)
    else:
        if os.path.exists(lesion_gt):
            gt_sitk = sitk.ReadImage(str(lesion_gt))
            gt = sitk.GetArrayFromImage(gt_sitk)
            if gt.shape != pred.shape:
                print(
                    f'WARNING: ground truth segmentation has a different '
                    f'shape than image: {lesion_gt}.\nAdjusting from '
                    f'{gt.shape} to {pred.shape}')
                gt = ndimage.zoom(
                    gt,
                    tuple(d1 / d2 for d1, d2 in zip(pred.shape, gt.shape)),
                    order=0
                )
                print(gt.shape, np.unique(gt))
        else:
            print(f"WARNING: non-existent ground truth file: {lesion_gt}")
            gt = np.zeros_like(pred)

        # Compute lesions volume
        lesion_volume_true = compute_volume(gt_sitk) if os.path.exists(lesion_gt) else np.nan

        # Calculate dice
        try:
            d = dice(gt, pred)
            # Calculate TP, FP, TN, FN
            TP = float(np.sum(np.logical_and(pred == 1, gt == 1)))
            TN = float(np.sum(np.logical_and(pred == 0, gt == 0)))
            FP = float(np.sum(np.logical_and(pred == 1, gt == 0)))
            FN = float(np.sum(np.logical_and(pred == 0, gt == 1)))

            FPR = ignore_zeros(FP, FP, TN)
            FNR = ignore_zeros(FN, FN, TP)
            FDR = ignore_zeros(FP, FP, TP) # this is false discovery rate, and I think this is what bianca calculates and calls FPR (in bianca_overlap_measures)

            NPV = ignore_zeros(TN, TN, FN)
            PPV = ignore_zeros(TP, TP, FP)
            sens = ignore_zeros(TP, TP, FN)
            spec = ignore_zeros(TN, TN, FP)

            FPR_biancaStyle = FP/(np.sum(pred == 1))
            FNR_biancaStyle = FN/(np.sum(gt == 1))
        except ValueError as ve:
            print(f'For {p}: {ve}\n{lesion_pred}\n{lesion_gt}')
            d = 'shape_error'
            FPR, FNR, FDR, NPV, PPV, sens, spec = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Print results to screen and return them
        return (p, d, FPR, FNR, FDR, NPV, PPV, sens, spec, lesion_volume_pred, lesion_volume_true)

def compute_stats_dir(gt_dir, pred_dir, outfile, num_processes=1, inference_only=False):
    patients = [p.split(".nii.gz")[0].split('_binary')[0] for p in os.listdir(pred_dir) if p.endswith(".nii.gz")]
    stats = run_multiprocessing(
        compute_stats, 
        patients, 
        fixed_arguments={'gt_dir':gt_dir, 'pred_dir':pred_dir, 'inference_only': inference_only},
        title="Computing stats",
        num_processes=num_processes)
    if inference_only:
        columns=('Patient', 'Predicted lesion volume')
    else:
        columns=('Patient', 'd', 'FPR', 'FNR', 'FDR', 'NPV', 'PPV', 'Sensitivity', 'Specificity', 'Predicted lesion volume', 'True lesion volume')
    stats = pd.DataFrame(stats, columns=columns).set_index('Patient')
    stats.to_csv(outfile)
    print("\nStats calculations completed.")
    return stats
