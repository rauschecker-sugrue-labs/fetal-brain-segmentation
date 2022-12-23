# This is a tensorflow implementation of elastic transformation for data augmentation
import numpy as np
import tensorflow as tf
from .LinearInterpolation import trilinear_sampler, trinnsampler

#%% 

class elastic_param():
    def __init__(self):
        # Affine parameters
        # Rotation, specify the maximum random rotation in radians
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        
        # Translation, specify the maximum random translation in normalized distance (0-1 to 0-X/Y/Z)
        self.trans_x = 0
        self.trans_y = 0
        self.trans_z = 0
        
        # scaling, specify the maximum random scaling in normalized size (1 + scale)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_z = 1.0
        
        # shearing, unimeplemented due to ins medical image the organ is usually kept unsheared
        
        # deformation parameters
        # Voxel shifting, normalized by scale
        self.df_x = 0.0
        self.df_y = 0.0
        self.df_z = 0.0
        
# input:
#    img: image, tensor of size (H, W, D, C)
#    seg: segmentation, tensor of size (H, W, D)
#    affine_param: the affine transformation parameter
#    elastic_param: the elastic transformation parameter
# output: 
#    out: the augmented image

def elastic_transform_3D(img, seg, elastic_param):
    # Compose Rotation, scaling, translation
    H = tf.shape(img)[0]
    W = tf.shape(img)[1]
    D = tf.shape(img)[2]
    C = tf.shape(img)[3]
    
    Hf = tf.cast(H, dtype = tf.float32)
    Wf = tf.cast(W, dtype = tf.float32)
    Df = tf.cast(D, dtype = tf.float32)
    Cf = tf.cast(C, dtype = tf.float32)
    
    # Generate homogenous grid
    y = tf.linspace(0.0, Hf-1, H)
    x = tf.linspace(0.0, Wf-1, W)
    z = tf.linspace(0.0, Df-1, D)
    x_t, y_t, z_t = tf.meshgrid(x, y, z)
    
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    z_t_flat = tf.reshape(z_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([y_t_flat, x_t_flat, z_t_flat, ones])
    sampling_grid = tf.cast(sampling_grid, tf.float32)   # cast to float32 (required for matmul)

    # Random rotation, random scaling, random translation
    rx = tf.random_uniform(shape = [], minval=-elastic_param.rotation_x, maxval= elastic_param.rotation_x, dtype=tf.float32)
    ry = tf.random_uniform(shape = [], minval=-elastic_param.rotation_y, maxval= elastic_param.rotation_y, dtype=tf.float32)
    rz = tf.random_uniform(shape = [], minval=-elastic_param.rotation_z, maxval= elastic_param.rotation_z, dtype=tf.float32)
    
    tx = Wf * tf.random_uniform(shape = [], minval=-elastic_param.trans_x, maxval= elastic_param.trans_x, dtype=tf.float32)
    ty = Hf * tf.random_uniform(shape = [], minval=-elastic_param.trans_y, maxval= elastic_param.trans_y, dtype=tf.float32)
    tz = Df * tf.random_uniform(shape = [], minval=-elastic_param.trans_z, maxval= elastic_param.trans_z, dtype=tf.float32)
    
    sx = tf.random_uniform(shape = [], minval=1-elastic_param.scale_x, maxval= 1+elastic_param.scale_x, dtype=tf.float32)
    sy = tf.random_uniform(shape = [], minval=1-elastic_param.scale_y, maxval= 1+elastic_param.scale_y, dtype=tf.float32)
    sz = tf.random_uniform(shape = [], minval=1-elastic_param.scale_z, maxval= 1+elastic_param.scale_z, dtype=tf.float32)
    
    # Form the affine matrix
    Ry = tf.stack([ (1.0, 0.0, 0.0, 0.0), (0.0, tf.cos(ry), -tf.sin(ry), 0.0), (0.0, tf.sin(ry), tf.cos(ry), 0.0), (0.0, 0.0, 0.0, 1.0)], axis=0)
    Rx = tf.stack([ (tf.cos(rx), 0.0, tf.sin(rx), 0.0), (0.0, 1, 0.0, 0.0), (-tf.sin(rx), 0.0, tf.cos(rx), 0.0), (0.0, 0.0, 0.0, 1.0)], axis=0)
    Rz = tf.stack([ (tf.cos(rz), -tf.sin(rz), 0.0, 0.0), (tf.sin(rz), tf.cos(rz), 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)], axis=0)
    R = tf.matmul(Rz, tf.matmul(Rx, Ry))
    S = tf.stack( [(sy, 0.0, 0.0 , 0.0), (0.0, sx, 0.0, 0.0), (0.0, 0.0, sz, 0.0), (0.0, 0.0, 0.0, 1.0)], axis = 0)
    T = tf.stack( [(1.0, 0.0, 0.0, ty), (0.0, 1.0, 0.0, tx), (0.0, 0.0, 1.0, tz), (0.0, 0.0, 0.0, 1.0)], axis = 0)
    
    # Affine transform
    A = tf.matmul(R, S)
    A = tf.matmul(A, T)
    yt_A, xt_A, zt_A, _ = tf.unstack(tf.matmul(A, sampling_grid))
    
    # Elastic transform
    xt_A = xt_A + tf.random_uniform( [H * W * D] , minval = -elastic_param.df_x, maxval = elastic_param.df_x)
    yt_A = yt_A + tf.random_uniform( [H * W * D] , minval = -elastic_param.df_y, maxval = elastic_param.df_y)
    zt_A = zt_A + tf.random_uniform( [H * W * D] , minval = -elastic_param.df_z, maxval = elastic_param.df_z)
    
    xt_A = tf.reshape(xt_A, [H, W, D])
    yt_A = tf.reshape(yt_A, [H, W, D])
    zt_A = tf.reshape(zt_A, [H, W, D])
    
    xt_A = tf.expand_dims(xt_A, axis = 0)
    yt_A = tf.expand_dims(yt_A, axis = 0)
    zt_A = tf.expand_dims(zt_A, axis = 0)
    
    # Interpolation
    img_ex = tf.expand_dims(img, axis = 0)
    img_A = trilinear_sampler(img_ex, xt_A, yt_A, zt_A, normalized_coordinate = False)
    img_A = tf.squeeze(img_A)
    
    seg_ex = tf.expand_dims(seg, axis = 0)
    seg_ex = tf.expand_dims(seg_ex, axis = -1)
    seg_A = trinnsampler(seg_ex, xt_A, yt_A, zt_A, normalized_coordinate = False)
    seg_A = tf.squeeze(seg_A)
    
    return img_A, seg_A, y_t, x_t, z_t

def elastic_transform_3D_tf2(img, seg, elastic_param):
    # Compose Rotation, scaling, translation
    H = tf.shape(input=img)[0]
    W = tf.shape(input=img)[1]
    D = tf.shape(input=img)[2]
    C = tf.shape(input=img)[3]
    
    Hf = tf.cast(H, dtype = tf.float32)
    Wf = tf.cast(W, dtype = tf.float32)
    Df = tf.cast(D, dtype = tf.float32)
    Cf = tf.cast(C, dtype = tf.float32)
    
    # Generate homogenous grid
    y = tf.linspace(0.0, Hf-1, H)
    x = tf.linspace(0.0, Wf-1, W)
    z = tf.linspace(0.0, Df-1, D)
    x_t, y_t, z_t = tf.meshgrid(x, y, z)
    
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    z_t_flat = tf.reshape(z_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([y_t_flat, x_t_flat, z_t_flat, ones])
    sampling_grid = tf.cast(sampling_grid, tf.float32)   # cast to float32 (required for matmul)

    # Random rotation, random scaling, random translation
    rx = tf.random.uniform(shape = [], minval=-elastic_param.rotation_x, maxval= elastic_param.rotation_x, dtype=tf.float32)
    ry = tf.random.uniform(shape = [], minval=-elastic_param.rotation_y, maxval= elastic_param.rotation_y, dtype=tf.float32)
    rz = tf.random.uniform(shape = [], minval=-elastic_param.rotation_z, maxval= elastic_param.rotation_z, dtype=tf.float32)
    
    tx = Wf * tf.random.uniform(shape = [], minval=-elastic_param.trans_x, maxval= elastic_param.trans_x, dtype=tf.float32)
    ty = Hf * tf.random.uniform(shape = [], minval=-elastic_param.trans_y, maxval= elastic_param.trans_y, dtype=tf.float32)
    tz = Df * tf.random.uniform(shape = [], minval=-elastic_param.trans_z, maxval= elastic_param.trans_z, dtype=tf.float32)
    
    sx = tf.random.uniform(shape = [], minval=1-elastic_param.scale_x, maxval= 1+elastic_param.scale_x, dtype=tf.float32)
    sy = tf.random.uniform(shape = [], minval=1-elastic_param.scale_y, maxval= 1+elastic_param.scale_y, dtype=tf.float32)
    sz = tf.random.uniform(shape = [], minval=1-elastic_param.scale_z, maxval= 1+elastic_param.scale_z, dtype=tf.float32)
    
    # Form the affine matrix
    Ry = tf.stack([ (1.0, 0.0, 0.0, 0.0), (0.0, tf.cos(ry), -tf.sin(ry), 0.0), (0.0, tf.sin(ry), tf.cos(ry), 0.0), (0.0, 0.0, 0.0, 1.0)], axis=0)
    Rx = tf.stack([ (tf.cos(rx), 0.0, tf.sin(rx), 0.0), (0.0, 1, 0.0, 0.0), (-tf.sin(rx), 0.0, tf.cos(rx), 0.0), (0.0, 0.0, 0.0, 1.0)], axis=0)
    Rz = tf.stack([ (tf.cos(rz), -tf.sin(rz), 0.0, 0.0), (tf.sin(rz), tf.cos(rz), 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)], axis=0)
    R = tf.matmul(Rz, tf.matmul(Rx, Ry))
    S = tf.stack( [(sy, 0.0, 0.0 , 0.0), (0.0, sx, 0.0, 0.0), (0.0, 0.0, sz, 0.0), (0.0, 0.0, 0.0, 1.0)], axis = 0)
    T = tf.stack( [(1.0, 0.0, 0.0, ty), (0.0, 1.0, 0.0, tx), (0.0, 0.0, 1.0, tz), (0.0, 0.0, 0.0, 1.0)], axis = 0)
    
    # Affine transform
    A = tf.matmul(R, S)
    A = tf.matmul(A, T)
    yt_A, xt_A, zt_A, _ = tf.unstack(tf.matmul(A, sampling_grid))
    
    # Elastic transform
    xt_A = xt_A + tf.random.uniform( [H * W * D] , minval = -elastic_param.df_x, maxval = elastic_param.df_x)
    yt_A = yt_A + tf.random.uniform( [H * W * D] , minval = -elastic_param.df_y, maxval = elastic_param.df_y)
    zt_A = zt_A + tf.random.uniform( [H * W * D] , minval = -elastic_param.df_z, maxval = elastic_param.df_z)
    
    xt_A = tf.reshape(xt_A, [H, W, D])
    yt_A = tf.reshape(yt_A, [H, W, D])
    zt_A = tf.reshape(zt_A, [H, W, D])
    
    xt_A = tf.expand_dims(xt_A, axis = 0)
    yt_A = tf.expand_dims(yt_A, axis = 0)
    zt_A = tf.expand_dims(zt_A, axis = 0)
    
    # Interpolation
    img_ex = tf.expand_dims(img, axis = 0)
    img_A = trilinear_sampler(img_ex, xt_A, yt_A, zt_A, normalized_coordinate = False)
    img_A = tf.squeeze(img_A)
    
    seg_ex = tf.expand_dims(seg, axis = 0)
    seg_ex = tf.expand_dims(seg_ex, axis = -1)
    seg_A = trinnsampler(seg_ex, xt_A, yt_A, zt_A, normalized_coordinate = False)
    seg_A = tf.squeeze(seg_A)
    
    return img_A, seg_A

def elastic_transform_2D(img, affine_param, elastic_param):
    pass
# %%
