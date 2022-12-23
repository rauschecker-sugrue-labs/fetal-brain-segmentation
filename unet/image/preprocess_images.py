##### Preprocess images #####
import os
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool, cpu_count

from utils.datasplit import read_split
from .elastic_transform_tf import elastic_transform_3D_tf2, elastic_param
from .patch_util import multi_resolution_patcher_3D
from .preprocess_Unet import save_patches_to_tfrecord

#TODO: watch resources used here by different steps
tf.debugging.set_log_device_placement(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only tf ERRORS are logged

#what can be done faster for inference only? --> no tfrecord maybe?

def img_to_tfrecord(config, train=True, val=True, test=False):
    global c
    c = config

    # Load elastic-transform parameters
    ep = elastic_transform_load(c.ep)
    trainDirs, valDirs, testDirs = read_split(split_file = c.train_test_csv, im_dir=c.data_nh_1mm)

    if train and len(trainDirs)>0: process_and_save(trainDirs, 'train', ep, c)
    if val and len(valDirs)>0: process_and_save(valDirs,   'val',   ep, c)
    if test and len(testDirs)>0: process_and_save(testDirs,  'test',  ep, c)

def process_and_save(images_info, TVT, elastic_param, c):
    train = TVT=='train'
    num_images = len(images_info)
    # Shuffle images
    if TVT in ['train', 'val']:
        trIdx = list(range(num_images))
        np.random.shuffle(trIdx)
        images_info = [ images_info[idx] for idx in trIdx ]
    num_images_batch = int(np.ceil(num_images / c.model_params.nTrainPerTfrecord))
    loop_tb = tqdm(range(num_images_batch), desc=f"Processing batch of {TVT} images", position=0)
    for tb in loop_tb:
        # Taking c.model_params.nTrainPerTfrecord images
        ia = tb * c.model_params.nTrainPerTfrecord
        ib = min((tb + 1) * c.model_params.nTrainPerTfrecord, num_images)
        # Generate patches #TODO: how to optimize numthreads here?
        patches = generate_patches(
            images_info[ia:ib],
            is_training = train,
            num_pos=c.model_params.num_pos,
            num_neg=c.model_params.num_neg,
            augmentation=c.model_params.aug,
            elastic_param=elastic_param,
            num_processes=c.num_cpu)
        # Save patches to TfRecords
        if c.model_params.nTrainPerTfrecord > 1:
            tfrecord_path = f"{c.tfrDir}/{TVT}_{tb:03d}.tfrecords"
        else:
            accession = images_info[ia][1]
            tfrecord_path = f"{c.tfrDir}/{TVT}_{accession}.tfrecords"
        save_patches_to_tfrecord(patches, tfrecord_path)



def generate_patches(
    images_info,
    is_training = True,
    num_pos=30,
    num_neg=30,
    augmentation=1,
    elastic_param=None,
    num_processes = cpu_count()):
    """
    Returns: patches list (image, seg, cpts, shape, disease, patient ID)
    """
    # Set multi-threading
    p = Pool(num_processes)
    num_images = len(images_info)
    num_images_thread = int(np.ceil(num_images/num_processes))
    patch_list = []
    loop_batch = tqdm(range(num_images_thread), desc=f" Generating patches for batch of images ({num_images})", position=1, leave=False)
    for i in loop_batch:
        images_batch_thread = images_info[i * num_processes : min((i+1) * num_processes, num_images)]
        # Parallelize the process of loading images
        loop_in_batch = tqdm(total=0, desc="  Loading images", position=2, leave=False)
        image_seg_pairs = p.map(load_image_seg_pair, images_batch_thread)
        image_seg_pairs_aug = []
        # Image augmentation (none in case of testing)
        loop_in_batch = tqdm(image_seg_pairs, desc=f"  Augmenting (x{augmentation})", position=2, leave=False)
        if not is_training:
            # Normalization #TODO: how does that affect volume calculation?
            for pair in loop_in_batch:
                image_np, seg_np, patient, disease = pair
                image_np = image_np.astype(np.float32)
                image_np = normalize_image(image_np)
                image_seg_pairs_aug.append((image_np, seg_np, patient, disease, is_training, num_pos, num_neg))
        else:
            for pair in loop_in_batch:
                image_np_orig, seg_np_orig, patient, disease = pair
                # Augmentation
                for i in range(augmentation):
                    if augmentation == 1: # No augmentation
                        image_np, seg_np = image_np_orig, seg_np_orig
                    else:
                        image_np_orig = tf.cast(image_np_orig, tf.float32)
                        image_np, seg_np = elastic_transform_3D_tf2(image_np_orig[:,:,:,None], seg_np_orig, elastic_param) #TODO: remove tf2
                    # Normalization
                    image_np = tf.cast(image_np, tf.float32).numpy()
                    image_np = normalize_image(image_np)
                    image_seg_pairs_aug.append((image_np, seg_np, patient, disease, is_training, num_pos, num_neg))
        # Generate patches
        # if is_training:
        #     patches_batch = p.map(crop_patches, image_seg_pairs_aug)
        # else:
        patches_batch = []
        loop_img_aug = tqdm(image_seg_pairs_aug, desc="  Cropping patches", position=2, leave=False)
        for pair in loop_img_aug:
            patches_batch.append(crop_patches(pair, c.model_params.patchsize_multi_res, c.model_params.test_patch_spacing, c.model_params.segsize))
        # Append to the global patch list
        for pat in patches_batch:
            patch_list.extend(pat)     
    # Close thread pool
    p.close()
    p.join()
    return patch_list

def load_image_seg_pair(image_info):
    """
    Input:      List with (image dir path, patient ID, disease)
    Returns:    SimpleITK image, segmentation, patient ID, disease
    """
    image_dir, patient, disease = image_info
    image_path  = f"{image_dir/patient}_1x1x1.nii.gz".replace(' ','')
    seg_path    = f"{image_dir/patient}_seg_1x1x1.nii.gz".replace(' ','')
    # Read image and segmentation
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    # Handle cases where the segmentation image is missing by inputing 0s
    if os.path.exists(seg_path):
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    else:
        seg= np.zeros_like(image, dtype = np.uint8)
    return image, seg, patient, disease

def normalize_image(image):
    """
    Normalizes an image
    """
    # Remove all possible artifact
    image[np.isnan(image)] = 0
    image[np.abs(image) > 1e308] = 0
    # Normalize the image
    image_voxels = image[image>0] # Get rid of the background before normalization
    image_norm = (image - np.mean(image_voxels)) / np.std(image_voxels)
    image_norm[image==0] = 0 # Put the background back where it was
    return image_norm

def crop_patches(image_info, patchsize_multi_res, test_patch_spacing, segsize):
    """
    Input: List with (image, seg, patient ID, disease, is_training, num_pos, num_neg), patchsize_multi_res, test_patch_spacing, segsize
    Returns: patches for one image (image, seg, cpts, shape, disease, patient ID)
    """
    image_np, seg_np, patient, disease, is_training, num_pos, num_neg = image_info
    # Crop patches
    if is_training:
        patches_pos_multi_res, patches_neg_multi_res =\
        multi_resolution_patcher_3D(image_np[:, :, :, None], seg_np, patchsize_multi_res, is_training = is_training, num_pos = num_pos, num_neg = num_neg)
        # Fit the patch to deepmedic format
        if disease == "BG_normal" or disease == "normal":
            patch_pos = []
        else:
            patch_pos = patch_to_Unet_format(patches_pos_multi_res, segsize, disease, patient)
        patch_negative = patch_to_Unet_format(patches_neg_multi_res, segsize, disease, patient)
        return patch_pos + patch_negative
    else:
        # Fit the patch to deepmedic format
        patches_multi_res = multi_resolution_patcher_3D(image_np[:, :, :, None], seg_np, patchsize_multi_res, is_training = is_training, spacing = test_patch_spacing)
        patches_multi_res = patch_to_Unet_format(patches_multi_res, segsize, disease, patient)
        return patches_multi_res

def patch_to_Unet_format(patches_multi_res, seg_size, disease, patient):
    """
    This function converts the patch from patch generator function to one that deepmedic needs
    """
    patches = []
    patches_high_res = patches_multi_res[0][0]
    for ph in patches_high_res:
        seg = ph[1]
        cpts = ph[2]
        shape = ph[3]
        patches.append((ph[0], seg, cpts, shape, disease, patient))
    return patches

def elastic_transform_load(param_dict):
    ep = elastic_param()
    for key, value in param_dict.items():
        setattr(ep, key, value)
    return ep
