#%% Preprocess script for deepmedic
import os
import numpy as np
import SimpleITK as sitk
# import tensorflow.compat.v1 as tf 
import tensorflow as tf #doens't work if version 2+ of tf is installed
from multiprocessing import Pool, cpu_count
import csv
import shutil
from tqdm.auto import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only tf ERRORS are logged
# tf.disable_v2_behavior() # cf. above
# tf.compat.v1.disable_eager_execution()

from .patch_util import multi_resolution_patcher_3D
from .elastic_transform_tf import elastic_transform_3D
from utils.datasplit import read_split

#%% Deepmedic requires
# input: 25^3 high resolution input and 19^3 low resolution
# output: 9^3 in the center in high resolution for segmentation

# This function converts the patch from patch generator function to one that deepmedic needs
def patch_to_Unet_format(patches_multi_res, seg_size, disease, patient):
    patches = []
    patches_high_res = patches_multi_res[0][0]
    
    for ph in patches_high_res:
        seg = ph[1]
        cpts = ph[2]
        shape = ph[3]
        patches.append((ph[0], seg, cpts, shape, disease, patient))
    return patches

# This function define normalization to image
def normalize_image(image_np):
    # Remove all possible artifact
    image_np[np.isnan(image_np)] = 0
    image_np[np.abs(image_np) > 1e308] = 0
    
    # Normalize the image
    image_voxels = image_np[image_np!=0] # Get rid of the background
    image_np_norm = (image_np - np.mean(image_voxels)) / np.std(image_voxels)
    image_np_norm[image_np==0] = 0
    
    return image_np_norm

def tf_elastic_define_graph():
    image_ph = tf.compat.v1.placeholder(tf.float32, shape = [None, None, None, None])
    seg_ph = tf.compat.v1.placeholder(tf.float32, shape = [None, None, None])
    image_aug, seg_aug, _, _, _ = ett.elastic_transform_3D(image_ph, seg_ph, c.ep)
    return image_ph, seg_ph, image_aug, seg_aug
    
# This function is a wrapper function for the tensorflow elastic transformation for data augmentation
# input:
#   image_np, seg_np: input image and segmentation, numpy array
def tf_elastic_wrapper(image_np, seg_np, sess, ops):
    image_ph, seg_ph, image_aug, seg_aug = ops
    image_aug_np, seg_aug_np = sess.run([image_aug, seg_aug], feed_dict = {image_ph:image_np[:,:,:,None], seg_ph:seg_np})
    return image_aug_np, seg_aug_np

# Auxiliary function for loading image and segmentation
def loadImageSegPair(dirDiseaseTuple):
    patientDir, patient, disease = dirDiseaseTuple
    
    # Load the image and segmentation
    #imageDir = patientDir + "FLAIR/FLAIR_1x1x1.nii.gz"
    #segDir = patientDir + "FLAIR/ManSeg_1x1x1.nii.gz"
    imageDir = patientDir + "%s_1x1x1.nii.gz" % patient
    segDir = patientDir + "%s_seg_1x1x1.nii.gz" % patient
    
    
    imageDir = imageDir.replace(' ', '')
    segDir = segDir.replace(' ', '')
    
    
    # Read in image and segmentation
    image_np_orig = sitk.GetArrayFromImage(sitk.ReadImage(imageDir))
    
    # Handle cases where the segmentation image is missing
    if os.path.exists(segDir):
        seg_np_orig = sitk.GetArrayFromImage(sitk.ReadImage(segDir))
    else:
        seg_np_orig= np.zeros_like(image_np_orig, dtype = np.uint8)
    
    return image_np_orig, seg_np_orig, patient, disease

# Auxiliary function for cropping patches
def cropPatches(ispairAndParam):
    image_np, seg_np, patient, disease, is_training, num_pos, num_neg = ispairAndParam
    
    # Crop patches
    if is_training:
        patches_pos_multi_res, patches_neg_multi_res =\
        multi_resolution_patcher_3D(image_np[:, :, :, None], seg_np, c.model_params.patchsize_multi_res, is_training = is_training, num_pos = num_pos, num_neg = num_neg)
        
        # Fit the patch to deepmedic format
        if disease == "BG_normal" or disease == "normal":
            patch_pos = []
        else:
            patch_pos = patch_to_Unet_format(patches_pos_multi_res, c.model_params.segsize, disease, patient)
        patch_negative = patch_to_Unet_format(patches_neg_multi_res, c.model_params.segsize, disease, patient)
        
        return patch_pos + patch_negative
    else:
        # Fit the patch to deepmedic format
        patches_multi_res = multi_resolution_patcher_3D(image_np[:, :, :, None], seg_np, c.model_params.patchsize_multi_res, is_training = is_training, spacing = c.model_params.test_patch_spacing)
        patches_multi_res = patch_to_Unet_format(patches_multi_res, c.model_params.segsize, disease, patient)
        return patches_multi_res
    
# This function generate patches according to deep medic format given list of directory.
# This is a very time consuming/very memory heavy function to run. So threadpool is used to optimized 
# performance
# for batch of directories:
#   load image (multi-thread) -> augmentation (single thread) -> patching (multi-thread)
def generate_deepmedic_patches(directories, is_training = True, num_pos = 100, num_neg = 100, aug = 1, ops = None, sess = None, num_thread = 1):
    p = Pool(num_thread)
    num_im = len(directories)
    num_batch = int(np.ceil(num_im /num_thread))
    
    # Generate patches
    patches = []
    loop_patch = tqdm(range(num_batch), desc=" Generating patches", position=1, leave=False)
    for i in loop_patch:
        directories_batch = directories[i * num_thread : min((i+1) * num_thread, num_im)]
        
        # Parallelize the process of loading image
        loop_in_patch = tqdm(total=0, desc="  Loading image", position=2, leave=False)
        image_seg_pairs = p.map(loadImageSegPair, directories_batch)
        image_seg_pairs_aug = []
        
        # No augmentation in test case
        loop_in_patch = tqdm(image_seg_pairs, desc="  Augmenting", position=2, leave=False)
        if not is_training:
            for ispair in loop_in_patch:
                image_np, seg_np, patient, disease = ispair
                image_np = image_np.astype(np.float32)
                image_np = normalize_image(image_np)
                image_seg_pairs_aug.append((image_np, seg_np, patient, disease, is_training, num_pos, num_neg))
        # Augment the image in training case - since it uses gpu, only serial execution
        else:
            for ispair in loop_in_patch:
                image_np_orig, seg_np_orig, patient, disease = ispair
            
                for j in range(aug):
                    if aug == 1: # No augmentation
                        image_np, seg_np = image_np_orig, seg_np_orig
                    else:
                        image_np, seg_np = tf_elastic_wrapper(image_np_orig, seg_np_orig, sess, ops)
                    # Normalize the image
                    image_np = image_np.astype(np.float32)
                    image_np = normalize_image(image_np)
                    image_seg_pairs_aug.append((image_np, seg_np, patient, disease, is_training, num_pos, num_neg))
            
        # Parallelize the process of generating patches
        if False:       
        #if is_training:       
            patches_batch = p.map(cropPatches, image_seg_pairs_aug)
        else:
            # It seems that the pool use pickle to , which has a limitation on how large an object it can 
            # pass out as return. So in test case it is saver to just use a for loop instead of pool
            # Shit! 
            patches_batch = []
            loop_in_patch = tqdm(image_seg_pairs_aug, desc="  Cropping patches", position=2, leave=False)
            for ispair in loop_in_patch:
                patches_batch.append(cropPatches(ispair))
        
        # Append to the global one
        for pat in patches_batch:
            patches.extend(pat)
    
    # Close the thread pool
    p.close()
    p.join()
    
    return patches


# This function combine image normalization, patch generation and save to tf record
# Need to do shuffle on patch level to ensure it is truly random. 
# This costs memory
def save_patches_to_tfrecord(patches, tfrecordDir):
    # shuffle the patches #TODO shouldn't we only do that for training?
    trIdx = list(range(len(patches)))
    np.random.shuffle(trIdx)
    
    f = tfrecordDir 
    writer = tf.io.TFRecordWriter(f)
    
    # Save the patches to tfrecord
    # iterate over each example
    # wrap with tqdm for a progress bar
    loop_trIdx = tqdm(trIdx, desc=" Saving patches to tfrecord", position=1, leave=False)
    for example_idx in loop_trIdx:
        p = patches[example_idx]
        cpts = p[2]
        shape = p[3]
        disease = p[4]
        patient = p[5]
    
        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'patch_high_res': tf.train.Feature(
                    bytes_list= tf.train.BytesList(value=[p[0].astype(np.float32).tostring()])),
                'seg': tf.train.Feature(
                    bytes_list= tf.train.BytesList(value=[p[1].astype(np.uint8).tostring()])),
                'phr_x': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[p[0].shape[0]])),
                'phr_y': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[p[0].shape[1]])),
                'phr_z': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[p[0].shape[2]])),
                'phr_c': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[p[0].shape[3]])),
                'seg_x': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[p[1].shape[0]])),
                'seg_y': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[p[1].shape[1]])),
                'seg_z': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[p[1].shape[2]])),
                'x': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[cpts[0]])),
                'y': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[cpts[1]])),
                'z': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[cpts[2]])),
                'h': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[shape[0]])),
                'w': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[shape[1]])),
                'd': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[shape[2]])),
                # 'disease': tf.train.Feature(
                #     int64_list=tf.train.Int64List(value=[c.diseaseCode[disease]])),
                        
                'patient': _bytes_feature(bytes(patient, encoding='ascii')), # python 3
                    # tf.train.Feature(
                    # int64_list=tf.train.Int64List(value=[int(patient)]))
        }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    
    writer.close()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

