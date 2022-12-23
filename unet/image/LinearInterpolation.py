#%% This is differentiable linear interpolation code implemented in Tensorflow. 
# It is based on the the implementation by Kelvin Zakka in Google, the author
# of spatial transformer network.
# The github link for spatial transformer network is here.
# https://github.com/kevinzakka/spatial-transformer-network

#%%
import tensorflow as tf

#%%

def get_pixel_value_2D(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: tensor of shape (B, H, W)
    - y: flattened tensor of shape (B, H, W)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], axis = 3)

    return tf.gather_nd(img, indices)

def get_pixel_value_3D(img, x, y, z):
    """
    Utility function to get pixel value for coordinate
    vectors x, y and z from a 5D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, D, C)
    - x: tensor of shape (B, H, W, D)
    - y: tensor of shape (B, H, W, D)
    - z: tensor of shape (B, H, W, D)

    Returns
    -------
    - output: tensor of shape (B, H, W, D, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = shape[3]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, (1, height, width, depth))

    indices = tf.stack([b, y, x, z], axis = 4)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, x, y, normalized_coordinate = False):
    """
    Performs bilinear sampling of the input images. Note that the 
    sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - interpolated images according to grids. Same size as grid.

    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    if normalized_coordinate:
        x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value_2D(img, x0, y0)
    Ib = get_pixel_value_2D(img, x0, y1)
    Ic = get_pixel_value_2D(img, x1, y0)
    Id = get_pixel_value_2D(img, x1, y1)
    
    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def trilinear_sampler(img, x, y, z, normalized_coordinate = False):
    """
    Performs trilinear sampling of the input images. Note that the 
    sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, D, C) layout.
    - grid: x, y, z the sampling grid on the image

    Returns
    -------
    - interpolated images according to grids. Same size as grid.

    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    D = tf.shape(img)[3]
    C = tf.shape(img)[4]

    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    max_z = tf.cast(D - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    # rescale x and y to [0, W/H]
    if normalized_coordinate:
        # cast indices as float32 (for rescaling)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        z = tf.cast(y, tf.float32)
        x = 0.5 * ((x + 1.0) * tf.cast(W, tf.float32))
        y = 0.5 * ((y + 1.0) * tf.cast(H, tf.float32))
        z = 0.5 * ((z + 1.0) * tf.cast(D, tf.float32))

    # grab 4 nearest corner points for each (x_i, y_i. z_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), tf.int32)
    z1 = z0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    # get pixel value at corner coords
    Ia_0 = get_pixel_value_3D(img, x0, y0, z0)
    Ia_1 = get_pixel_value_3D(img, x0, y0, z1)
    Ib_0 = get_pixel_value_3D(img, x0, y1, z0)
    Ib_1 = get_pixel_value_3D(img, x0, y1, z1)
    Ic_0 = get_pixel_value_3D(img, x1, y0, z0)
    Ic_1 = get_pixel_value_3D(img, x1, y0, z1)
    Id_0 = get_pixel_value_3D(img, x1, y1, z0)
    Id_1 = get_pixel_value_3D(img, x1, y1, z1)
    
    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)
    z0 = tf.cast(z0, tf.float32)
    z1 = tf.cast(z1, tf.float32)

    # calculate deltas
    wa_0 = (x1-x) * (y1-y) * (z1-z)
    wa_1 = (x1-x) * (y1-y) * (z-z0)
    wb_0 = (x1-x) * (y-y0) * (z1-z)
    wb_1 = (x1-x) * (y-y0) * (z-z0)
    wc_0 = (x-x0) * (y1-y) * (z1-z)
    wc_1 = (x-x0) * (y1-y) * (z-z0)
    wd_0 = (x-x0) * (y-y0) * (z1-z)
    wd_1 = (x-x0) * (y-y0) * (z-z0)

    # add dimension for addition
    wa_0 = tf.expand_dims(wa_0, axis=4)
    wa_1 = tf.expand_dims(wa_1, axis=4)
    wb_0 = tf.expand_dims(wb_0, axis=4)
    wb_1 = tf.expand_dims(wb_1, axis=4)
    wc_0 = tf.expand_dims(wc_0, axis=4)
    wc_1 = tf.expand_dims(wc_1, axis=4)
    wd_0 = tf.expand_dims(wd_0, axis=4)
    wd_1 = tf.expand_dims(wd_1, axis=4)

    # compute output
    out = tf.add_n([wa_0*Ia_0, wb_0*Ib_0, wc_0*Ic_0, wd_0*Id_0, \
                    wa_1*Ia_1, wb_1*Ib_1, wc_1*Ic_1, wd_1*Id_1])

    return out

# Nearest neighbor sampler
def binnsampler(img, x, y, normalized_coordinate = False):
    """
    Performs 2D nearest neigbour sampling of the input images. Note 
    that the sampling is done identically for each channel of the 
    input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - interpolated images according to grids. Same size as grid.

    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    if normalized_coordinate:
        x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
        y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(x, 'int32')
    y0 = tf.cast(y, 'int32')

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)

    # get pixel value at corner coords
    out = get_pixel_value_2D(img, x0, y0)

    return out

# Nearest neighbor sampler
def trinnsampler(img, x, y, z, normalized_coordinate = False):
    """
    Performs 3D nearest neigbour sampling of the input images. Note 
    that the sampling is done identically for each channel of the 
    input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - interpolated images according to grids. Same size as grid.

    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    D = tf.shape(img)[3]
    C = tf.shape(img)[4]

    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    max_z = tf.cast(D - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    # rescale x and y to [0, W/H]
    if normalized_coordinate:
        # cast indices as float32 (for rescaling)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        z = tf.cast(z, tf.float32)
        x = 0.5 * ((x + 1.0) * tf.cast(W, tf.float32))
        y = 0.5 * ((y + 1.0) * tf.cast(H, tf.float32))
        z = 0.5 * ((z + 1.0) * tf.cast(D, tf.float32))

    # grab 4 nearest corner points for each (x_i, y_i. z_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(x, tf.int32)
    y0 = tf.cast(y, tf.int32)
    z0 = tf.cast(z, tf.int32)

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)

    # get pixel value at corner coords
    out = get_pixel_value_3D(img, x0, y0, z0)
    
    return out


