###### Model definition ######
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import sys

default_kernel_size = (3, 3, 3)
OUTPUT_CHANNELS = 1

down_res_block_filters = [(16, 16, 64, 2), (64, 64, 128, 2), (128, 128, 256, 3), (256, 256, 256, 3)]
up_res_block_filters = [(256, 256, 128, 1), (128, 128, 64, 1), (64, 64, 16, 1), (16, 16, 16, 1)]
final_filters = (16, 16, 1)

#### Model Functions ####

# A wrapper around the normalization layer
def norm_layer(name):
    return tf.keras.layers.BatchNormalization(name = name + "_bn")

# Convolution sandwich building block
def sandwich(filters, kernel_size, dilation, name):
    l = tf.keras.Sequential(name=name)
    l.add(tf.keras.layers.Conv3D(filters, kernel_size, dilation_rate=dilation, activation= tf.nn.relu, name=name + "_conv", padding='same'))
    l.add(norm_layer(name))
    return l


# The residual block
def resblock(X, filters, filters2, kernel_size, dilation, name):
    # The long way
    l_lw = tf.keras.Sequential(name=name+"_lgWay")
    l_lw.add(tf.keras.layers.Conv3D(filters, kernel_size, dilation_rate=dilation, activation= tf.nn.relu, name=name + "_sand1_conv", padding='same'))
    l_lw.add(norm_layer(name+"_sand1"))
    # l_lw.add(sandwich(filters, default_kernel_size, dilation, name = name + "_sand1"))
    l_lw.add(tf.keras.layers.Conv3D(filters2, kernel_size, dilation_rate=dilation, activation= tf.nn.relu, name=name + "_sand2_conv", padding='same'))
    l_lw.add(norm_layer(name+"_sand2"))
    # l_lw.add(sandwich(filters2, default_kernel_size, dilation, name = name + "_sand2"))
    
    # The short way
    l_sw = tf.keras.Sequential(name=name+"_shWay")
    l_sw.add(tf.keras.layers.Conv3D(filters2, (1, 1, 1), name=name + "_short_conv", padding='same'))
    l_sw.add(norm_layer(name = name + "_short"))
    
    l = tf.keras.layers.Add(name=name+"_resAdd")([l_lw(X), l_sw(X)])
    return l

# Downsampling block
def downsampling_block(filters, name):
    l = tf.keras.Sequential(name=name)
    l.add(tf.keras.layers.Conv3D(filters = filters, strides = (2, 2, 2), kernel_size = default_kernel_size, padding = 'same', activation = tf.nn.relu))
    l.add(norm_layer(name))
    return l

# Upsampling block
def upsampling_block(filters, name):
    l = tf.keras.Sequential(name=name)
    l.add(tf.keras.layers.Conv3DTranspose(filters = filters, strides=(2, 2, 2), kernel_size = default_kernel_size, padding = 'same', activation = tf.nn.relu))
    l.add(norm_layer(name))
    return l



#### Metric and losses ####

class DiceLoss(tf.keras.metrics.Metric):
    """tried to define a DiceLoss class -- haven't got it to work yet
    """
    def __init__(self, name="dice_loss", eps=1e-6, **kwargs):
        super(DiceLoss, self).__init__(name=name, **kwargs)
        self.__name__ = name
        self.eps = eps
        self.dice_loss = self.add_weight(name="dice", initializer="zeros")

    def update_state(self, label, logits, sample_weight=None):
        label = tf.cast(label, tf.float32)
        logits = tf.cast(logits, tf.float32)
        intersection = tf.reduce_sum(logits * label, axis = (1, 2, 3))
        union = tf.reduce_sum(logits, axis = (1, 2, 3)) + tf.reduce_sum(label, axis = (1, 2, 3))
        loss =  2.0*(intersection + self.eps)/ (union + 2.0*self.eps)
        self.dice_loss.assign_add(tf.reduce_mean(-loss))

    def result(self):
        return self.dice_loss

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dice_loss.assign(0.0)

class DiceScore(tf.keras.metrics.Metric):
    def __init__(self, name="dice_score", eps=1e-6, **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.__name__ = name
        self.eps = eps
        self.dice_score = self.add_weight(name="dice", initializer="zeros")

    def update_state(self, label, logits, sample_weight=None):
        label = tf.cast(label, tf.float32)
        logits = tf.cast(logits, tf.float32)
        intersection = tf.reduce_sum(logits * label, axis = (1, 2, 3))
        union = tf.reduce_sum(logits, axis = (1, 2, 3)) + tf.reduce_sum(label, axis = (1, 2, 3))
        score =  2.0*(intersection + self.eps)/ (union + 2.0*self.eps)
        self.dice_score.assign_add(tf.reduce_mean(score))

    def result(self):
        return self.dice_score

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dice_score.assign(0.0)


def soft_dice(_type="score", smooth = 1e-5):
    """ Computes Dice
    Args: 
        _type: "score" or "loss"
        smooth: smoothing coefficient
    Returns:
        either Dice score or Dice loss (1-score)
    """
    def dice_score(y_true, y_pred):
        # Flatten
        y_true_f = K.cast(K.flatten(y_true), y_pred.dtype)
        y_pred_f = K.flatten(y_pred)
        # Sum
        im_sum = K.sum(y_true_f) + K.sum(y_pred_f)
        im_sum = K.cast(im_sum, tf.float32)
        # Intersection
        intersection = K.sum(y_true_f * y_pred_f)
        intersection = K.cast(intersection, tf.float32)
        # Return Dice coefficient
        return (2. * intersection + smooth) / (im_sum + smooth)
    
    def dice_loss(y_true, y_pred):
        return 1-dice_score(y_true, y_pred)

    if _type == "score":
        return dice_score
    elif _type == "loss":
        return dice_loss

def weighted_cross_entropy(weight_alpha=0.9, binary=True):
    def _loss(y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype)
        weights = y_true * (weight_alpha/(1.-weight_alpha)) + 1.
        bce = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        # axis = (1, 2, 3, 4) if binary else (1, 2, 3)
        weighted_loss = K.mean(bce * weights)
        return weighted_loss
    return _loss

def focal_loss(alpha=0.25, gamma=2.0):
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy
    return SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)


#### Model architecture definition ####

def UNet(kernel_size=default_kernel_size,
         down_res_block_filters=down_res_block_filters,
         up_res_block_filters=up_res_block_filters,
         final_filters=final_filters, input_shape = (96,96,96,1), training=True): #TODO: check training=True/False
    X_input = tf.keras.Input(input_shape)
    X = X_input

    # Down
    X_inter = []
    for k, filters_tuple in enumerate(down_res_block_filters):
        filters, filters2, filter_downsample, dilation = filters_tuple
        X = resblock(X, filters, filters2, kernel_size, dilation, name = "dblock_%d_res_down"%k)
        X_inter.append(X)
        X = downsampling_block(filter_downsample, name = "dblock_%d_down"%k)(X)

    # Reverse the intermediary blocks to send to upsampling
    X_inter = X_inter[::-1]

    # Up
    K = len(up_res_block_filters)
    for k, (filters_tuple, inter) in enumerate(zip(up_res_block_filters, X_inter)):
        filters, filters2, filter_upsample, dilation = filters_tuple
        X = resblock(X, filters, filters2, kernel_size, dilation, name = "ublock_%d_res_up"%(K-k-1))
        X = upsampling_block(filter_upsample, name = "ublock_%d_up"%(K-k-1))(X)
        X = tf.keras.layers.Concatenate(axis = -1, name="ublock_%d_concat"%(K-k-1))([X, inter])

    # Final layer
    for k, filters in enumerate(final_filters):
        X = sandwich(filters, (1, 1, 1), 1, name = "final_sandwich_%d"%k)(X)
    X_output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='prediction')(X)
    
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = X_output, name = "UNet")

    return model

class UNetModel(tf.keras.Model):
    """ Class defining the training steps for the model
    Args:
        x, y: model input tensor, model output
    """
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

def UNet_custom_train(kernel_size=default_kernel_size,
                      down_res_block_filters=down_res_block_filters,
                      up_res_block_filters=up_res_block_filters,
                      final_filters=final_filters,
                      input_shape = (96,96,96,1),
                      training=True): #TODO: check training=True/False

    X_input = tf.keras.Input(input_shape)
    X = X_input

    # Down
    X_inter = []
    for k, filters_tuple in enumerate(down_res_block_filters):
        filters, filters2, filter_downsample, dilation = filters_tuple
        X = resblock(X, filters, filters2, kernel_size, dilation, name = "dblock_%d_res_down"%k)
        X_inter.append(X)
        X = downsampling_block(filter_downsample, name = "dblock_%d_down"%k)(X)

    # Reverse the intermediary blocks to send to upsampling
    X_inter = X_inter[::-1]

    # Up
    K = len(up_res_block_filters)
    for k, (filters_tuple, inter) in enumerate(zip(up_res_block_filters, X_inter)):
        filters, filters2, filter_upsample, dilation = filters_tuple
        X = resblock(X, filters, filters2, kernel_size, dilation, name = "ublock_%d_res_up"%(K-k-1))
        X = upsampling_block(filter_upsample, name = "ublock_%d_up"%(K-k-1))(X)
        X = tf.keras.layers.Concatenate(axis = -1, name="ublock_%d_concat"%(K-k-1))([X, inter])

    # Final layer
    for k, filters in enumerate(final_filters):
        X = sandwich(filters, (1, 1, 1), 1, name = "final_sandwich_%d"%k)(X)
    X = tf.keras.activations.sigmoid(X)

    # Create model, using the custom training loop
    # model = UNetModel(X_input, X)

    return X_input, X

#### Freezing layers ####
def freeze_layers(model, instring, all_but=True, debug=True):
    """
    Freeze layers that contain `instring` in their name (or all but that layer with `all_but` argument)
        for example, instring='ublock' freezes all up blocks.
        see model.summary() for layer names
    """
    def freeze(elem, all_but):
        for key, layer in layer_dict.items():
            if all_but:
                freeze_layer = elem not in key
            else:
                freeze_layer = elem in key
            if freeze_layer:
                layer.trainable = False
                if debug: print(f"Froze: {key}")
    
    layer_dict = {l.name: model.get_layer(l.name) for l in model.layers}
    
    if type(instring) == list:
        if all_but:
            sys.exit("Freezing layers currently doesn't support freezing all but a list of layers.")
        else:
            for elem in instring:
                freeze(elem, all_but)
    elif type(instring) == str:
        freeze(instring, all_but)
    else:
        print("wrong input")
        exit
    
    if debug: model.summary()


