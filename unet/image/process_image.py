### Process images (both preprocess and postprocess)
import os
import shutil

import numpy as np
import scipy
import SimpleITK as sitk
from tqdm.auto import tqdm
from utils.multiprocessing import run_multiprocessing

from .fix_orientation import fixOrientation, getImSpDirct, reverseOrientation
from .resample_util import resample_im_by_spacing, resample_seg_by_spacing


def image_already_processed(image, output_dir): #TODO it doesn't work
    # don't process images that have already been!
    # check output directory for existence
    if image in os.listdir(output_dir):
        return True
    return False

def create_image_list(input_dir, image_list):
    if image_list is None:
        images = [p for p in os.listdir(input_dir) if p.endswith(".nii.gz")]
    else:
        image_list = list(map(str, image_list))
        images = []
        for im in os.listdir(input_dir):
            # extract patient ID from image name
            if im.endswith("_seg.nii.gz"):
                p = im.split("_seg.nii.gz")[0]
            elif im.endswith(".nii.gz"):
                p = im.split(".nii.gz")[0]
            # check if it's in the list provided
            if p in image_list:
                images.append(im)
    return images

##Strip out the header
def strip_header_dir(data_raw_dir, data_nh_dir, image_list=None, num_processes=1, force_preprocess=False):
    images = create_image_list(data_raw_dir, image_list)
    run_multiprocessing(strip_header,
                        images,
                        fixed_arguments={'input_dir':data_raw_dir,
                                         'output_dir':data_nh_dir,
                                         'force_preprocess':force_preprocess},
                        num_processes=num_processes,
                        title="Stripping header")
    return

def strip_header(im_name, input_dir, output_dir, force_preprocess):
    if not force_preprocess and image_already_processed(im_name, output_dir): return
    im_in = input_dir / im_name
    im_out = output_dir / im_name
    try:
        im = sitk.ReadImage(str(im_in))
        im_np, spacing, direction = getImSpDirct(im)
        im_np, spacing = fixOrientation(im_np, direction, spacing)
        im = sitk.GetImageFromArray(im_np)
        im.SetSpacing(list(spacing))
        sitk.WriteImage(im, str(im_out))
    except Exception as err:
        print(f'Error in strip_header for {im_name}.\n {type(err)}\n {err.args}')
    # except ValueError as err:
    #     print(f'Error in strip_header for {im_name}.\n {type(err)}\n {err.args}')
    return

##Resample the image and segmentation into 1x1x1 mm
def resample_to_1mm_dir(data_nh_dir, data_nh_1mm, image_list=None, num_processes=1, force_preprocess=False):
    images = create_image_list(data_nh_dir, image_list)
    run_multiprocessing(resample_to_1mm,
                        images,
                        fixed_arguments={'input_dir':data_nh_dir,
                                         'output_dir':data_nh_1mm,
                                         'force_preprocess':force_preprocess},
                        num_processes=num_processes,
                        title="Resampling images")

def resample_to_1mm(im, input_dir, output_dir, force_preprocess):
    if "native" in im:
        return
    if not force_preprocess and image_already_processed(im.split(".nii.gz")[0]+"_1x1x1.nii.gz", output_dir): return
    im_in = input_dir / im
    im_out = output_dir / im
    
    # Read the image and segmentation
    if "seg" in im:
        seg_sitk = sitk.ReadImage(str(im_in))
        seg_np = sitk.GetArrayFromImage(seg_sitk)
        spacing = seg_sitk.GetSpacing()[::-1]
        seg_re_np = resample_seg_by_spacing(seg_np, spacing)
        sitk.WriteImage( sitk.GetImageFromArray(seg_re_np), str(im_out).split(".nii.gz")[0] + "_1x1x1.nii.gz")
    else:
        image_sitk = sitk.ReadImage(str(im_in))
        image_np = sitk.GetArrayFromImage(image_sitk)
        image_np = image_np[:, :, :, None]
        spacing = image_sitk.GetSpacing()[::-1]
        image_re_np = resample_im_by_spacing(image_np, spacing)
        image_re_np = np.squeeze(image_re_np)
        sitk.WriteImage( sitk.GetImageFromArray(image_re_np), str(im_out).split(".nii.gz")[0] + "_1x1x1.nii.gz")
    return

##Resample the validation output to original space
# Resize the segmentation to original space and reverse the orientation

def resample_from_1mm_dir(c, num_processes=1):
    patients = [p.split(".nii.gz")[0] for p in os.listdir(c.valoutDir) if p.endswith(".nii.gz")]
    run_multiprocessing(resample_from_1mm,
                        patients,
                        fixed_arguments={'input_dir':c.valoutDir,
                                        'output_dir':c.resampledDir,
                                        'noheader_dir':c.data_nh,
                                        'raw_dir':c.data_raw},
                        num_processes=num_processes,
                        title="Resampling images")

def resample_from_1mm(accession, input_dir, output_dir, noheader_dir, raw_dir):
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(input_dir / (accession + ".nii.gz"))))
    noheader = sitk.ReadImage(str(noheader_dir / (accession + ".nii.gz")))
    header = sitk.ReadImage(str(raw_dir / (accession + ".nii.gz")))
    
    # Resampling
    old_size = noheader.GetSize()[::-1]
    pred_old = scipy.ndimage.zoom(pred, (old_size[0]/pred.shape[0], old_size[1]/pred.shape[1], old_size[2]/pred.shape[2] ), order = 1)  
    
    if (pred_old.shape[0]!=old_size[0]) or (pred_old.shape[1]!=old_size[1]) or (pred_old.shape[2]!=old_size[2]):
        print(accession)
    
    # Reverse direction
    direction = np.array(header.GetDirection()).reshape((3, 3))
    pred_old_reverse = reverseOrientation(pred_old, direction)
    
    # Copy header and spacing
    out = sitk.GetImageFromArray(pred_old_reverse)
    out.SetSpacing(noheader.GetSpacing())
    out.CopyInformation(header)
    
    sitk.WriteImage(out, str(output_dir / (accession + ".nii.gz")))

#%% Run binarization of images

def binarize(patient, prob_seg_dir, bin_seg_dir, threshold, fill_holes=False):
    import nibabel as nib
    prob_seg_file =  prob_seg_dir  / (str(patient) + ".nii.gz")
    md = bin_seg_dir / str(threshold)
    md.mkdir(parents=True, exist_ok=True)
    bin_seg_file = md / (str(patient) + "_binary.nii.gz")
    # im_sitk = sitk.ReadImage(str(prob_seg_file))
    # im_np = sitk.GetArrayFromImage(im_sitk)
    img_nib = nib.load(prob_seg_file)
    img_np = img_nib.get_fdata()
    seg_np = np.where(img_np > threshold, 1, 0)
    if fill_holes:
        seg_np = scipy.ndimage.binary_fill_holes(seg_np).astype(int)
    nib.save(nib.Nifti1Image(seg_np,img_nib.affine,img_nib.header), bin_seg_file)
    # sitk.WriteImage(sitk.GetImageFromArray(seg_np).CopyInformation(im_sitk), bin_seg_file)
    # command = f"fslmaths {prob_seg_file} -thr {threshold} -bin {bin_seg_file}"
    # os.system(command)

def binarize_dir(input_dir, output_dir, threshold, num_processes=1, fill_holes=False):
    patients = [p.split(".nii.gz")[0] for p in os.listdir(input_dir) if p.endswith(".nii.gz")]
    # if type(threshold) is not list: threshold = [threshold]
    run_multiprocessing(binarize,
                        patients,
                        fixed_arguments={'prob_seg_dir':input_dir,
                                        'bin_seg_dir':output_dir,
                                        'threshold':threshold,
                                        'fill_holes':fill_holes},
                        num_processes=num_processes,
                        title="Binarization")
