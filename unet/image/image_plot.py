from tensorflow.keras.callbacks import Callback
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

class ImageHistory(Callback):
    def __init__(self, tensorboard_dir, data, num_images_to_show=10):
        super(ImageHistory, self).__init__()
        self.tensorboard_dir = str(tensorboard_dir)
        self.batches_for_plot = self.get_random_batches(data, num_images_to_show)
        # get indices for a few batches, and keep them so we can see the progression on the same images
        ind_bright, ind_dimm, ind_random = [], [], []
        for _, batch_seg in self.batches_for_plot:
            ind_bright.append(find_best_indices(find_brightest, batch_seg))
            ind_dimm.append(find_best_indices(find_dimmest, batch_seg))
            ind_random.append(find_best_indices(find_random, batch_seg))
        self.indices = ind_bright, ind_dimm, ind_random # axes: (num_func, num_batch, ind_tuple)
        self.indices = np.moveaxis(self.indices, 1, 0)  # axes: (num_batch, num_func, ind_tuple)

    def on_epoch_end(self, epoch, logs={}):
        recap_images = []
        for bb, batch_imgseg in enumerate(self.batches_for_plot):
            batch_img, batch_seg = batch_imgseg
            batch_pred = self.model.predict(batch_imgseg) #TODO batch_size?
            # Get `best` 2D slices
            b_ind = self.indices[bb]
            bright = batch_img[b_ind[0][0], b_ind[0][1], :, :], batch_seg[b_ind[0][0], b_ind[0][1], :, :], batch_pred[b_ind[0][0], b_ind[0][1], :, :]
            dimm   = batch_img[b_ind[1][0], b_ind[1][1], :, :], batch_seg[b_ind[1][0], b_ind[1][1], :, :], batch_pred[b_ind[1][0], b_ind[1][1], :, :]
            rand   = batch_img[b_ind[2][0], b_ind[2][1], :, :], batch_seg[b_ind[2][0], b_ind[2][1], :, :], batch_pred[b_ind[2][0], b_ind[2][1], :, :]
            # Display them in a grid
            figure = image_grid([bright, dimm, rand], title=['Bright', 'Dimm', 'Random'], figsize=(4.7,5))
            # Transforms figure into Tensor
            recap_image = plot_to_image(figure)
            recap_images.append(recap_image)
        _, h, w, c = recap_image.shape
        recap_images = np.reshape(recap_images, (-1, h, w, c))
        writer = tf.summary.create_file_writer(self.tensorboard_dir)
        with writer.as_default():
            tf.summary.image("Images and segmentations after each epoch", recap_images, max_outputs=len(recap_images), step=epoch) #(value=[tf.Summary.Value(tag='Images and segmentations', image=recap_images)])
        return

    def get_random_batches(self, dataset, num_image_to_show, rand_span=10):
        range_batch = rand_span * num_image_to_show
        rand_batches = list(range(range_batch))
        random.shuffle(rand_batches)
        rand_batches = rand_batches[:num_image_to_show]
        batches_for_plot = []
        for bb, batch in enumerate(dataset.repeat().take(range_batch)):
            if bb in rand_batches:
                batches_for_plot.append(batch)
        return batches_for_plot


def image_grid(imgsegpred_list, title=None, figsize=(4.7,5)):
    """Creates figure with list of triples (img2d, seg2d, pred2)
    """
    n = len(imgsegpred_list)
    if not title: title=['']*n
    # Create a figure to contain the plot.
    fig, axes = plt.subplots(n,3, sharex=True, sharey=True, figsize=figsize)
    for ax, triple, ylabel in zip(axes, imgsegpred_list, title):
        ax[0].imshow(triple[0], cmap=plt.cm.binary, aspect='equal')
        if ylabel!='': ax[0].set_ylabel(ylabel, fontsize=10)
        ax[1].imshow(triple[1], cmap=plt.cm.binary, aspect='equal')
        ax[2].imshow(triple[2], cmap=plt.cm.binary, aspect='equal')
    ax[0].set_title('Image', y=-0.3, fontsize=10)
    ax[1].set_title('Segmentation', y=-0.3, fontsize=10)
    ax[2].set_title('Prediction', y=-0.3, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.xticks([])
    plt.yticks([])
    return fig

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def find_brightest(img):
    img = np.squeeze(img)
    brightest = 0
    i_brightest = 0
    for i in range(img.shape[0]):
        bright = np.sum(img[i])
        if bright > brightest:
            brightest = bright
            i_brightest = i
    return i_brightest, brightest

def find_dimmest(img):
    img = np.squeeze(img)
    dimmest = float('inf')
    i_dimmest = 0
    for i in range(img.shape[0]):
        bright = np.sum(img[i])
        if bright < dimmest:
            dimmest = bright
            i_dimmest = i
    return i_dimmest, dimmest

def find_random(img):
    img = np.squeeze(img)
    i_rand = random.randint(0, img.shape[0]-1)
    return i_rand, np.sum(img[i_rand])

def find_best_indices(func, batch_segs):
    """ Return the batch and 3D indices corresponding to the brightest/dimmest (whatever func chooses) true segmentation
    """
    # find image in batch
    best_img_from_batch_i, _ = func(batch_segs)
    # find slice in 3D
    best_slice_i, slice_brightness = func(batch_segs[best_img_from_batch_i])
    return best_img_from_batch_i, best_slice_i

