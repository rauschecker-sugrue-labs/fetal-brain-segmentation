#%% This is a script that fix the orientation of nifti image
import SimpleITK as sitk
import numpy as np

# This helper function get the direction, spacing and numpy array from an SimpleITK image
# input:
#   im: a SimpleITK image
# output:
#   im_np, spacing, direction: the image, spacing, direction, all converted to numpy array. Note that
#   the axis of the image numpy array is flipped backward in this (Z, Y, X).
def getImSpDirct(im):
    direction = np.array(im.GetDirection()).reshape((3, 3))
    spacing = np.array(im.GetSpacing())
    im_np = sitk.GetArrayFromImage(im)
    
    return im_np, spacing, direction
    

# This function fix image orientation and swap spacing given direction
# input:
#   im_np, spacing, direction: the image, spacing, direction, all converted to numpy array. Note that
#   the axis of the image numpy array is flipped backward in this (Z, Y, X).
# output:
#   im_np, spacing: the image and spacing fixed by the direction. So these two can be straight up used to 
#   create a SimpleITK image with identity direction and it will look the same as the original. Note that
#   the im_np direction is again still flipped backward in (Z, Y, X).
def fixOrientation(im_np, direction, spacing):
    im_np = np.swapaxes(im_np, axis1 = 0, axis2 = 2) # From X Y Z to Z X Y
    ax_phy = np.argmax(np.abs(direction), axis = 1)
    sign_phy = np.sign(np.array( [ direction[0, ax_phy[0]], direction[1, ax_phy[1]], direction[2, ax_phy[2]] ]))
    
    # Transpose axis
    im_np = np.transpose(im_np, axes = ax_phy)
    sign_phy = sign_phy[ax_phy]
    spacing = spacing[ax_phy]
    
    # Flip axis
    for ax, sn in enumerate(list(sign_phy)):
        if sn == -1:
            im_np = np.flip(im_np, axis = ax)
            
    im_np = np.swapaxes(im_np, axis1 = 0, axis2 = 2)
    
    return im_np, spacing

# This function does the opposite of the fixOrientation. Given an image without the header and an image that does, this
# function reverse the orientation of the image without header
def reverseOrientation(im_np, direction):
    im_np = np.swapaxes(im_np, axis1 = 0, axis2 = 2) # From X Y Z to Z X Y
    ax_phy = np.argmax(np.abs(direction), axis = 1)
    sign_phy = np.sign(np.array( [ direction[0, ax_phy[0]], direction[1, ax_phy[1]], direction[2, ax_phy[2]] ]))
    sign_phy = sign_phy[ax_phy]
    
    # Reverse the conversion
    ax_im = np.array([np.argwhere(ax_phy==0)[0, 0], np.argwhere(ax_phy==1)[0, 0], np.argwhere(ax_phy==2)[0, 0]])
    sign_im = sign_phy[ax_im]
    
    # Transpose axis
    im_np = np.transpose(im_np, axes = ax_im)
    
     # Flip axis
    for ax, sn in enumerate(list(sign_im)):
        if sn == -1:
            im_np = np.flip(im_np, axis = ax)

    im_np = np.swapaxes(im_np, axis1 = 0, axis2 = 2)
    
    return im_np
    
