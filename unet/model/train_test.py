##### train/test/predict #####
import os
from pathlib import Path
import logging
from multiprocessing import Pool
from functools import partial
import tensorflow as tf
tf.get_logger().setLevel(logging.WARN)
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm
from utils.tic_toc import tic, toc
from .model_definition import UNet, soft_dice, weighted_cross_entropy, freeze_layers
from image.postprocess_deepmedic import assemSegFromPatches_dir
from image.preprocess_images import load_image_seg_pair, normalize_image, crop_patches
from image.image_plot import ImageHistory


def dataset_from_tfr(tfrDir, buffer_size, batch_size, test_train, img_list=None):
    """Load dataset from tensorflow records
    Args:
        tfrDir: directory for Tensorflow records
        buffer_size: buffer size for element shuffling
        batch_size: batch size
        test_train: "test" or "train" (different behaviors depending on which one is chosen - i.e. no shuffle for test time)
        img_list: if set, chooses a subsample of training records that matches the names provided in that list
    Returns:
        tf.Dataset
    """
    train = test_train=="train"
    if img_list is not None:
        tfrecords = [str(tr) for tr in tfrDir.iterdir() if tr.name.endswith(".tfrecords") and test_train in tr.stem and tr.stem.split('_',1)[1] in img_list]
        buffer_size = len(tfrecords)
    else:
        tfrecords = [str(tfrDir / tr) for tr in os.listdir(tfrDir) if tr.endswith(".tfrecords") and test_train in tr]
    try:
        ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=len(tfrecords))
    except:
        ds = tf.data.TFRecordDataset(tfrecords)
    if train:
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True) #seed=2020 TODO: check if loss function is repeating itself?
        ds = ds.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE) 
        if img_list is None: ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True) # shuffle twice since several examples per tfrecord
    elif test_train == "val":
        ds = ds.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def load_image_train(tfrecord):
    tfrecord_features = tf.io.parse_single_example(
        tfrecord,
        features={
            'patch_high_res': tf.io.FixedLenFeature([], tf.string),
            'seg': tf.io.FixedLenFeature([], tf.string),
            'phr_x': tf.io.FixedLenFeature([], tf.int64),
            'phr_y': tf.io.FixedLenFeature([], tf.int64),
            'phr_z': tf.io.FixedLenFeature([], tf.int64),
            'phr_c': tf.io.FixedLenFeature([], tf.int64),
            'seg_x': tf.io.FixedLenFeature([], tf.int64),
            'seg_y': tf.io.FixedLenFeature([], tf.int64),
            'seg_z': tf.io.FixedLenFeature([], tf.int64),
            'x': tf.io.FixedLenFeature([], tf.int64),
            'y': tf.io.FixedLenFeature([], tf.int64),
            'z': tf.io.FixedLenFeature([], tf.int64),
            #'disease': tf.io.FixedLenFeature([], tf.int64),
            'patient': tf.io.FixedLenFeature([], tf.string)
        }
    )
    
    phr_x, phr_y, phr_z, phr_c = tfrecord_features['phr_x'], tfrecord_features['phr_y'], tfrecord_features['phr_z'], tfrecord_features['phr_c']
    phr = tf.io.decode_raw(tfrecord_features['patch_high_res'], tf.float32)
    phr = tf.reshape(phr, shape = tf.stack([phr_x, phr_y, phr_z, phr_c]))
    
    seg_x, seg_y, seg_z = tfrecord_features['seg_x'], tfrecord_features['seg_y'], tfrecord_features['seg_z']
    seg = tf.io.decode_raw(tfrecord_features['seg'], tf.uint8)
    seg = tf.reshape(seg, shape = tf.stack([seg_x, seg_y, seg_z]))
    seg = tf.expand_dims(seg, axis = -1)
    
    # disease = tfrecord_features['disease']
    patient = tf.io.decode_raw(tfrecord_features['patient'], tf.uint8)
    
    sz = tf.size(patient)
    patient = tf.pad(patient, [[0, 100 - sz]])
    
    x, y, z = tfrecord_features['x'], tfrecord_features['y'], tfrecord_features['z']
    
    # Random flip left and right
    flip = tf.random.uniform([1], minval=0.0, maxval=1.0)
    flip = tf.squeeze(flip)
    phr = tf.cond(flip < 0.5, lambda: tf.reverse(phr, axis = [2]), lambda: phr)
    seg = tf.cond(flip < 0.5, lambda: tf.reverse(seg, axis = [2]), lambda: seg)

    return phr, seg #, x, y, z, patient

def load_image_test(tfrecord):
    tfrecord_features = tf.io.parse_single_example(
        tfrecord,
        features={
            'patch_high_res': tf.io.FixedLenFeature([], tf.string),
            'seg': tf.io.FixedLenFeature([], tf.string),
            'phr_x': tf.io.FixedLenFeature([], tf.int64),
            'phr_y': tf.io.FixedLenFeature([], tf.int64),
            'phr_z': tf.io.FixedLenFeature([], tf.int64),
            'phr_c': tf.io.FixedLenFeature([], tf.int64),
            'seg_x': tf.io.FixedLenFeature([], tf.int64),
            'seg_y': tf.io.FixedLenFeature([], tf.int64),
            'seg_z': tf.io.FixedLenFeature([], tf.int64),
            'x': tf.io.FixedLenFeature([], tf.int64),
            'y': tf.io.FixedLenFeature([], tf.int64),
            'z': tf.io.FixedLenFeature([], tf.int64),
            'h': tf.io.FixedLenFeature([], tf.int64),
            'w': tf.io.FixedLenFeature([], tf.int64),
            'd': tf.io.FixedLenFeature([], tf.int64),
            # 'disease': tf.io.FixedLenFeature([], tf.int64),
            'patient': tf.io.FixedLenFeature([], tf.string)
        }
    )
    
    phr_x, phr_y, phr_z, phr_c = tfrecord_features['phr_x'], tfrecord_features['phr_y'], tfrecord_features['phr_z'], tfrecord_features['phr_c']
    phr = tf.io.decode_raw(tfrecord_features['patch_high_res'], tf.float32)
    phr = tf.reshape(phr, shape = tf.stack([phr_x, phr_y, phr_z, phr_c]))
    
    seg_x, seg_y, seg_z = \
    tfrecord_features['seg_x'], tfrecord_features['seg_y'], tfrecord_features['seg_z']
    seg = tf.io.decode_raw(tfrecord_features['seg'], tf.uint8)
    seg = tf.reshape(seg, shape = tf.stack([seg_x, seg_y, seg_z]))
    seg = tf.expand_dims(seg, axis = -1)
    
    #    disease = tfrecord_features['disease']
    #    patient = tfrecord_features['patient']
    patient = tf.io.decode_raw(tfrecord_features['patient'], tf.uint8)
    sz = tf.size(patient)
    patient = tf.pad(patient, [[0, 100 - sz]])
    
    x, y, z = tfrecord_features['x'], tfrecord_features['y'], tfrecord_features['z']
    h, w, d = tfrecord_features['h'], tfrecord_features['w'], tfrecord_features['d']
    
    return phr, seg, x, y, z, h, w, d, patient

def loss_choice(loss_name='WCE', **loss_options):
    """ Returns the loss function based on string name
        Supported loss names:
            WCE: weighted cross entropy loss
            focal: focal loss ## not yet implemented
        Supported loss options:
            see individual losses arguments
    """
    if loss_name == 'WCE':
        if 'weight_alpha' in loss_options.keys():
            return weighted_cross_entropy(weight_alpha=loss_options['weight_alpha'])
        else:
            return weighted_cross_entropy()
    elif loss_name == 'focal':
        print('not yet implemented')
        exit 

def create_model(c, freeze=None, cpt=None):
    # Mixed precision training and dynamic loss scaling
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    optimizer = tf.keras.optimizers.Adam(learning_rate=c.model_params.learning_rate)
    # optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic") ## no need for that when using model.fit()

    # Enabling Accelerated Linear Algebra
    # tf.config.optimizer.set_jit(True)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = UNet()
        METRICS = [
                   # tf.keras.metrics.Precision(name='precision'),
                   # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                   # tf.keras.metrics.Recall(name='recall'),
                   soft_dice()
                  ]
        if freeze is not None:
            freeze_layers(model, instring=freeze)
        model.compile(optimizer=optimizer,
                      loss=loss_choice(c.model_params.training_loss, **c.model_params.loss_options),#soft_dice(_type="loss"), #tf.keras.losses.BinaryCrossentropy(from_logits=True), #weightedLoss(tf.keras.losses.BinaryCrossentropy(from_logits=True), {0:0.95, 1:0.05}),
                      metrics=METRICS)
        
        # Loading model within the scope of Distributed Strategy if applicable
        if c.load_model is not None:
            model.load_weights(c.load_model)
    
    return model

def train(train_data, c, notes=None, validation_data=None, freeze=None):
    # Create model
    model = create_model(c, freeze=freeze)

    # Model saving - checkpoint definition
    checkpoint_path = str(c.modelDir / "cp-{epoch:03d}.ckpt")
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq='epoch',
                                                     monitor='loss',
                                                     save_best_only=True)
    
    # TensorBoard setup
    logDir = c.tfboardDir / (c.timestamp + "_" + notes)
    print(f"\ntensorboard --logdir '{logDir}'\n")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logDir,
                                                #  histogram_freq=1,  # How often to log histogram visualizations
                                                #  embeddings_freq=1,  # How often to log embedding visualizations
                                                #  update_freq=30,
                                                #  profile_batch=10
                                                 )
    
    # Show images on TensorBoard
    image_history_callback = ImageHistory(tensorboard_dir=logDir/'images', data=train_data, num_images_to_show=c.model_params.num_images_to_show)

    progress_bar = 1 if c.interactive else 2
    # Start training
    model_history = model.fit(train_data,
                              initial_epoch=c.starting_epoch,
                              epochs=c.model_params.num_epochs+c.starting_epoch,
                            #   batch_size=c.model_params.batch_size,
                            #   steps_per_epoch=1000, #TODO what does that mean? -- previously while True loop to go through every example
                              callbacks=[cp_callback, tb_callback, ], #image_history_callback
                              validation_data=validation_data,
                              verbose=progress_bar) # progress bar (1), epoch only (2)
    
    # Save model
    tf.saved_model.save(model, str(c.modelDir))
    return model_history

def predict(test_dataset, c, presize=False):
    """ Predict the given dataset
    Args:
        test_dataset: tf.Dataset object
        c: config file
        presize: precompute the size of dataset to display progress bar correctly (adds an upfront cost)
    """
    # Load model
    print("Load model...")
    model = create_model(c)
    
    # Loop over the dataset to run prediction and save info
    if presize:
        num_batches=0
        print("Evaluating dataset size...")
        for batch in test_dataset:
            num_batches+=1
    else: 
        num_batches=None
    pred_list=[]
    from utils.tic_toc import tic, toc
    c.logging = {f'Predict_batch({c.model_params.batch_size})': []}
    loop_dataset = tqdm(test_dataset, total=num_batches, desc="Predicting batches", position=0)
    for batch in loop_dataset:
        tic()
        batch_size = batch[0].shape[0]
        pred_batch = model.predict(batch, batch_size=1)
        # loop over the batch size
        for id_b in range(batch_size):
            img, seg_ini = batch[0][id_b], batch[1][id_b]
            x, y, z, h, w, d, patient = batch[2][id_b].numpy(), batch[3][id_b].numpy(), batch[4][id_b].numpy(), batch[5][id_b].numpy(), batch[6][id_b].numpy(), batch[7][id_b].numpy(), batch[8][id_b].numpy(), 
            pred = pred_batch[id_b]
            pred_list.append((x, y, z, h, w, d, patient, pred))
        c.logging[f'Predict_batch({c.model_params.batch_size})'].append(toc(print_to_screen=False))
    # Assemble the prediction into complete output file and write those to a folder
    X = [p[0] for p in pred_list]
    Y = [p[1] for p in pred_list]
    Z = [p[2] for p in pred_list]
    cpts = np.stack([X, Y, Z], axis = 1)
    H = [p[3] for p in pred_list]
    W = [p[4] for p in pred_list]
    D = [p[5] for p in pred_list]
    shape = np.stack([H, W, D], axis = 1)
    patients = [p[6] for p in pred_list]
    patches_pred = [p[7] for p in pred_list]

    patients_str = []
    loop = tqdm(list(patients), desc="Cleaning up patient IDs", position=0)
    for p in loop:
        pstr = "".join([chr(c) for c in p]).replace("\x00", "")
        patients_str.append(pstr)

    patients_str = np.array(patients_str)
    patients_list = list(np.unique(patients_str))

    # Stitching patches and saving images
    c.logging['Stitch_preprocess'] = []
    c.logging['Stitch_assemble'] = []
    c.logging['Stitch_writetodisk'] = []
    loop = tqdm(patients_list, desc="Stitching patches and saving segmentation", position=0)
    for p in loop:
        tic()
        # Extract patches for certain patient
        idx = np.where(patients_str == p)[0]
        cpts_p = cpts[idx]
        shape_p = shape[idx]
        patches_pred = np.array(patches_pred)
        patches_pred_p = patches_pred[idx]
        c.logging['Stitch_preprocess'].append(toc(print_to_screen=False))
        # Assemble into segmentation and save to a file
        tic()
        seg = assemSegFromPatches_dir(shape_p[0], cpts_p, patches_pred_p)
        c.logging['Stitch_assemble'].append(toc(print_to_screen=False))
        tic()
        sitk.WriteImage(sitk.GetImageFromArray(seg), str(c.valoutDir / (p + ".nii.gz")))
        c.logging['Stitch_writetodisk'].append(toc(print_to_screen=False))


def predict_batch(c):
    from utils.datasplit import read_split
    train_list, val_list, test_list = read_split(split_file = c.train_test_csv, im_dir=c.data_nh_1mm)
    if c.predict_on == 'all':
        img_info = train_list + val_list + test_list
    elif c.predict_on == 'train':
        img_info = train_list
    elif c.predict_on == 'val':
        img_info = val_list
    elif c.predict_on == 'test':
        img_info = test_list
    img_paths = [k[0]/(k[1]+'_1x1x1.nii.gz') for k in img_info]
    # Instantiate model if not already
    tic()
    model = create_model(c)
    c.logging['Model loading'] = [toc(print_to_screen=False, restart=True)]
    pred_list = []
    seg_list = []
    c.logging['Load image'] = []
    c.logging['Normalize image'] = []
    c.logging['Create patches'] = []
    c.logging['Predict image'] = []
    c.logging['Assemble patches'] = []
    c.logging['Save image to disk'] = []
    for img_path in tqdm(img_paths, desc='Predicting images', position=0):
        try:
            pred, seg = predict_single(img_path, c, model)
            pred_list.append(pred)
            seg_list.append(seg)
        except Exception as err:
            print('Issue in prediction with', img_path)
            print(err)
            continue
    return pred_list, seg_list

def predict_single(img_path, c, model=None, condition='NA', ignore_existing=False):
    img_meta = ((img_path.parent), str(img_path.stem)[:-10], condition)
    # Load image based on path
    img, seg, patient, disease = load_image_seg_pair(img_meta)
    c.logging['Load image'].append(toc(print_to_screen=False, restart=True))
    out_path = c.valoutDir / (patient + ".nii.gz")
    if ignore_existing and out_path.is_file(): return

    # Instantiate model if not already
    if model is None: model = create_model(c)
    img = normalize_image(img)
    c.logging['Normalize image'].append(toc(print_to_screen=False, restart=True))
    
    # seg = normalize_image(seg)
    # Crop patches
    patches = crop_patches((img, seg, patient, disease, False, c.model_params.num_pos, c.model_params.num_neg), c.model_params.patchsize_multi_res, c.model_params.test_patch_spacing, c.model_params.segsize)
    patches = img_to_model_format(patches)
    img_to_predict = np.array(patches[0])
    c.logging['Create patches'].append(toc(print_to_screen=False, restart=True))

    # Predict
    patient_patch_prediction = model.predict(img_to_predict, batch_size=c.model_params.batch_size)
    c.logging['Predict image'].append(toc(print_to_screen=False, restart=True))

    # Reassemble the prediction
    pred = assemSegFromPatches_dir(patches[3][0][:3], patches[2], patient_patch_prediction, position=-1)
    pred = np.squeeze(pred)
    c.logging['Assemble patches'].append(toc(print_to_screen=False, restart=True))
    sitk.WriteImage(sitk.GetImageFromArray(pred), str(out_path))
    c.logging['Save image to disk'].append(toc(print_to_screen=False, restart=True))

    return pred, seg

def img_to_model_format(patches):
    img, seg, cpts, shape, disease, patient = [], [], [], [], [], []
    for patch in patches:
        img.append(patch[0])
        seg.append(patch[1])
        cpts.append(patch[2])
        shape.append(patch[3])
        disease.append(patch[4])
        patient.append(patch[5])
    p_img = [i for i in img]
    p_seg = [tf.cast(tf.expand_dims(seg, axis = -1), dtype=tf.uint8) for seg in seg] # change type to uint8
    p_cpts = [i for i in cpts]
    p_shape = [i for i in shape]
    p_disease = [i for i in disease]
    p_patient = [i for i in patient]
    return p_img, p_seg, p_cpts, p_shape, p_disease, p_patient