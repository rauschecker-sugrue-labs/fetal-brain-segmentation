#%% This scrips defines resample utility
import numpy as np
from scipy import ndimage
import tensorflow as tf

from .LinearInterpolation import trilinear_sampler, trinnsampler

# This function uses numpy nd image implementation of tri-linear interpolation to resample image
# input:
#   image: input image, [H, W, D, C]
#   seg: segmentation, [H, W, D]
#   res: 0-1, relative resolution of output to original image
# outputs:
#   image_resize, seg_resize: image, segmentation that get resampled
def resample_by_resolution(image, seg, res):
    image_resize = ndimage.zoom(image, (res, res, res, 1.0), order = 1)
    seg_resize = ndimage.zoom(seg, (res, res, res), order = 0)
    
    return image_resize, seg_resize

# This function generate grid by resolution
# input:
#   height, width, depth: Tensor, size of the image being resampled
#   res: list of length 3, each 0-1, relative resolution of output to original image in 3 dimension
def grid_generator_tf(height, width, depth, res):
    y = tf.linspace(0.0, tf.cast(height, tf.float32) - 1, res[0] * height)
    x = tf.linspace(0.0, tf.cast(width, tf.float32) - 1, res[1] * width)
    z = tf.linspace(0.0, tf.cast(depth, tf.float32) - 1, res[2] * depth)
    x_t, y_t, z_t = tf.meshgrid(x, y, z)
    
    x_t = tf.expand_dims(x_t, axis=0)
    y_t = tf.expand_dims(y_t, axis=0)
    z_t = tf.expand_dims(z_t, axis=0)
    
    return x_t, y_t, z_t


# This function uses tensorflow implementation of tri-linear interpolation to resample image
# input:
#   image: input image, Tensor, size [H, W, D, C]
#   seg: segmentation, Tensor, [H, W, D]
#   res: list of length 3, each 0-1, relative resolution of output to original image in 3 dimension
# outputs:
#   image_resize, seg_resize: Tensor, image, segmentation that get resampled
def resample_by_resolution_tf(image, seg, res):
    height, width, depth, C = tf.shape(image)
    x_t, y_t, z_t = grid_generator_tf(height, width, depth, res)
    
    image = tf.expand_dims(image, axis = 0)
    image_resize = trilinear_sampler(image, x_t, y_t, z_t)
    image_resize = tf.squeeze(image_resize)
    
    seg = tf.expand_dims(seg, axis = 0)
    seg = tf.expand_dims(seg, axis = -1)
    seg_resize = trinnsampler(seg, x_t, y_t, z_t)
    seg_resize = tf.squeeze(seg_resize)
    
    return image_resize, seg_resize

# This function uses tensorflow implementation of tri-linear interpolation to resample just segmentation
# input:
#   labels: labels in 3D, Tensor, [N, H, W, D, C]
#   res: resample resolution factor.
#   interp: interpolation method, either nn or linear. Default nn.
# outputs:
#   labels_resize: Tensor, [N ,H, W, D, C], segmentation that get resampled
def resample_labels_tf(labels, res, interp = "nn"):
    N, height, width, depth, C = tf.shape(labels)
    x_t, y_t, z_t = grid_generator_tf(height, width, depth, res)
    
    if interp == "nn":
        labels_resize = trinnsampler(labels, x_t, y_t, z_t)
    elif interp == "linear":
        labels_resize = trilinear_sampler(labels, x_t, y_t, z_t)
    else:
        raise ValueError("Invalid interp type in resamples")
        
    return labels_resize

# This function resampled img and seg to 1x1x1.
# inputs: 
#   img: the image, [H, W, D, C]
#   seg: the segmentation, [H, W, D]
#   spacing: tuple, voxel spacing in 3D before resampled
#   new_spacing: tuple, voxel spacing in 3D after resampled
# outputs:
#   image_resize, seg_resize: image, segmentation that get resampled
def resample_by_spacing(image, seg, spacing, new_spacing = (1.0, 1.0, 1.0)):
    # calculate resize factor
    rf = np.array(spacing).astype(np.float64) / np.array(new_spacing).astype(np.float64)
    image_resize = ndimage.zoom(image, (rf[0], rf[1], rf[2], 1.0), order = 1)
    seg_resize = ndimage.zoom(seg, rf, order = 0)
    return image_resize, seg_resize

def resample_im_by_spacing(image, spacing, new_spacing = (1.0, 1.0, 1.0)):
    # calculate resize factor
    rf = np.array(spacing).astype(np.float64) / np.array(new_spacing).astype(np.float64)
    image_resize = ndimage.zoom(image, (rf[0], rf[1], rf[2], 1.0), order = 1)
    return image_resize

def resample_seg_by_spacing(seg, spacing, new_spacing = (1.0, 1.0, 1.0)):
    # calculate resize factor
    rf = np.array(spacing).astype(np.float64) / np.array(new_spacing).astype(np.float64)
    seg_resize = ndimage.zoom(seg, rf, order = 0)
    return seg_resize


# This function resampled img and seg to 1x1x1 using tensorflow trilinear interpolation.
# inputs: 
#   image: input image, Tensor, size [H, W, D, C]
#   seg: segmentation, Tensor, [H, W, D]
#   spacing: tuple, voxel spacing in 3D before resampled
#   new_spacing: tuple, voxel spacing in 3D after resampled
# outputs:
#   image_resize, seg_resize: Tensor, image, segmentation that get resampled
def resample_by_spacing_tf(image, seg, spacing, new_spacing = (1.0, 1.0, 1.0)):
    # calculate resize factor
    rf = np.array(spacing).astype(np.float64) / np.array(new_spacing).astype(np.float64)
    image_resize, seg_resize = resample_by_resolution_tf(image, seg, rf)
    return image_resize, seg_resize