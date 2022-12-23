# This contains utility that generate patches from 3D volume
#%% Import libraries
import numpy as np
from .resample_util import resample_by_resolution

#%% Utility functions


# This function sample 3D patches from 3D volume
# input:
#   image: the image in numpy array, dimension [H, W, D, C] 
#   seg: segmentation of the image, dimension [H, W, D], right now assuming this is binary
#   patch_size: the size of patch
#   num_pos: number of positive patches that contains lesion to sample. If there is no enough patches to sample,
#            it will return all indexes that contains lesion
#   num_negative: number of negative background patches that doesn't contain lesion to sample.
# output:
#   patches_pos, patches_neg: list of (img_patch, seg_patch, cpt)
def single_resolution_patcher_3D(image, seg, patch_size, is_training = True, num_pos = 10, num_neg = 10, spacing = [1, 1, 1]):
    if is_training:
        # Randomly sample center points
        cpts_pos_sampled, cpts_neg_sampled = sample_center_points(seg, num_pos, num_neg)
        # Crop patches around center points
        patches_pos = crop_patch_by_cpts(image, seg, cpts_pos_sampled, patch_size)
        patches_neg = crop_patch_by_cpts(image, seg, cpts_neg_sampled, patch_size)
    
        return patches_pos, patches_neg
    else:
        # Regularly grid center points
        cpts = grid_center_points(image.shape, spacing)
        # Crop patches around center points
        patches = crop_patch_by_cpts(image, seg, cpts, patch_size)
        return patches
        

# This function sample 3D patches from 3D volume in multiple resolution around same center, used deepmedic or 
# similar style network
# input:
#   image: the image in numpy array, dimension [H, W, D, C] 
#   seg: segmentation of the image, dimension [H, W, D], right now assuming this is binary    
#   patchsize_multi_res: this is the patch size in multi-resolution [(1, (25, 25, 25)), (0.33, (19, 19, 19))]
#                   this means it will sample patch size (25, 25, 25) in resolution 1x, patch size (19, 19, 19) in resolution 0.33x etc    
#   num_pos: number of positive patches that contains lesion to sample. If there is no enough patches to sample,
#            it will return all indexes that contains lesion
#   num_negative: number of negative background patches that doesn't contain lesion to sample.
def multi_resolution_patcher_3D(image, seg, patchsize_multi_res, is_training = True, num_pos = 10, num_neg = 10, spacing = [1, 1, 1]):
    # Sample center points
    import tensorflow as tf
    seg = tf.cast(seg, dtype='int16')
    if is_training:
        cpts_pos_sampled, cpts_neg_sampled = sample_center_points(seg, num_pos, num_neg)
        
        # Get center pts in multi resolution
        cpts_pos_multi_res = multiple_resolution_cpts(cpts_pos_sampled, patchsize_multi_res)
        cpts_neg_multi_res = multiple_resolution_cpts(cpts_neg_sampled, patchsize_multi_res)
        
        patches_pos_multi_res = []
        patches_neg_multi_res = []
        for idx, pr in enumerate(patchsize_multi_res):
            res, patch_size = pr
            # Downsample the image and segmentation
            image_resize, seg_resize = resample_by_resolution(image, seg, res)
            
            cpts_max = np.array(image_resize.shape[:3]) - 1
            cpts_max = cpts_max[:, None]
            
            # Fetch positive patches
            cpts_pos = cpts_pos_multi_res[idx]
            cpts_pos = np.minimum(cpts_max, cpts_pos) # Limit the range
            # Due to numerical rounding the cpts in different resolution may not match the 
            # resize image exactly. So need to hard constraint it
            
            patches = crop_patch_by_cpts(image_resize, seg_resize, cpts_pos, patch_size)
            patches_pos_multi_res.append([patches, res])
            
            # Fetch positive patches
            cpts_neg = cpts_neg_multi_res[idx]
            cpts_neg = np.minimum(cpts_max, cpts_neg) # Limit the range.
            patches = crop_patch_by_cpts(image_resize, seg_resize, cpts_neg, patch_size)
            patches_neg_multi_res.append([patches, res])
        
        return patches_pos_multi_res, patches_neg_multi_res
    else:
        # Regularly grid center points
        cpts = grid_center_points(image.shape, spacing)
        cpts_multi_res = multiple_resolution_cpts(cpts, patchsize_multi_res)
        patches_multi_res = []
        
        for idx, pr in enumerate(patchsize_multi_res):
            res, patch_size = pr
            # Downsample the image and segmentation
            image_resize, seg_resize = resample_by_resolution(image, seg, res)
            
            # Fetch patches
            cpts_res = cpts_multi_res[idx]
            patches_res = crop_patch_by_cpts(image_resize, seg_resize, cpts_res, patch_size)
            patches_multi_res.append([patches_res, res])
            
        return patches_multi_res

# This function samples center points from segmentation for patching. 
# Implement all patch selection in this function. Leave other function clean
# input:
#   seg: segmentation of the image, dimension [H, W, D], right now assuming this is binary
#   num_pos: number of positive patches that contains lesion to sample. If there is no enough patches to sample,
#            it will return all indexes that contains lesion
#   num_negative: number of negative background patches that doesn't contain lesion to sample.
def sample_center_points(seg, num_pos, num_neg):
    idx_pos = np.stack(np.where(seg>0), axis = 0)
    
    if idx_pos[0].shape[0]<num_pos:
        cpts_pos_sampled = idx_pos
    else:
        idx_rand = np.random.choice(idx_pos[0].shape[0], num_pos, replace = False)
        cpts_pos_sampled = idx_pos[:, idx_rand]
        
    idx_neg = np.stack(np.where(seg==0), axis = 0)
    
    if idx_neg[0].shape[0]<num_neg:
        cpts_neg_sampled = idx_neg
    else:
        idx_rand = np.random.choice(idx_neg[0].shape[0], num_neg, replace = False)
        cpts_neg_sampled = idx_neg[:, idx_rand]
    
    return cpts_pos_sampled, cpts_neg_sampled


# This function generate center points in order of image. Just to keep the API consistent
def grid_center_points(shape, stride):
    x = np.arange(start = 0, stop = shape[0], step = stride[0])
    y = np.arange(start = 0, stop = shape[1], step = stride[1])
    z = np.arange(start = 0, stop = shape[2], step = stride[2])
    x_t, y_t, z_t = np.meshgrid(x, y, z, indexing = "ij")
    
    idx = np.stack([x_t.flatten(), y_t.flatten(), z_t.flatten()], axis = 0)
    
    return idx
    

# This function converts center points to multiple resolution
def multiple_resolution_cpts(cpts, patchsize_multi_res):
    cpts_multi_res = []
    for pr in patchsize_multi_res:
        res, _ = pr
        cpts_res = (cpts * res).astype(np.int32)
        cpts_multi_res.append(cpts_res)
    return cpts_multi_res

# This function crops patches around center points with patch size
# input:
#   image: input image, [H, W, D, C]
#   seg: segmentation, [H, W, D]
#   cpts: center points, [3, N]
#   patch_size: tuple, (HP, WP, DP)
# output:
#   patches: list of (img_patch, seg_patch, cpt)
def crop_patch_by_cpts(image, seg, cpts, patch_size):
    half_size = np.ceil((np.array(patch_size) - 1) / 2).astype(np.int32)
    N = cpts.shape[1]
    patches = []
    
    # Padded the image and segmentation here so that the out of boundary cases are taken care of
    image_padded = np.pad(image, ((half_size[0], half_size[0]), (half_size[1], half_size[1]), (half_size[2], half_size[2]), (0, 0)), mode = "constant", constant_values = 0)
    seg_padded = np.pad(seg, ((half_size[0], half_size[0]), (half_size[1], half_size[1]), (half_size[2], half_size[2])), mode = "constant", constant_values = 0)
    
    shape = image.shape
    
    for i in range(N):
        cpt = cpts[:, i]
        l = cpt
        u = cpt + np.array(patch_size)
        img_patch = image_padded[l[0]:u[0], l[1]:u[1], l[2]:u[2], :]
        seg_patch = seg_padded[l[0]:u[0], l[1]:u[1], l[2]:u[2]]
        patches.append((img_patch, seg_patch, cpt, shape))
        
        if (img_patch.shape[0] == 95) or (img_patch.shape[1] == 95) or (img_patch.shape[2] == 95):
            print ("Debugging ", cpt, l, u, image_padded.shape)
        
    return patches

    