#%% This script splits the dataset and saves the split into a csv file. This should be run before the 
# preprocessing script get run so that the split be read by the preprocessing script
import numpy as np
import csv
import os
import pandas as pd
from pathlib import Path

#%%
def create_cv_splits(num_folds: int, images_info_csv: Path, test_size=None) -> None:
    """ Create csv files for the number of folds with train/test examples chosen at random
    Parameters
    ----------
        num_folds: int
            number of folds for the cross validation. Corresponds to the number of csv files that are created.
        images_info_csv: importlib.Path
            csv file path containing the list of images. The fold csvs will be stored in the same location.
        test_size: float
            percent of images to reserve for the test set. It will be floored. If None, set to num_folds/num_images.
    """
    images_info = pd.read_csv(images_info_csv, names=['Patient', 'Condition', 'TT'])
    num_images = len(images_info)
    if test_size is None:
        num_test = num_images // num_folds
    elif not 0 <= test_size <= 1:
        print(f'Error: test_size must be between 0 and 1. {test_size} given.')
        return
    else:
        num_test = int(test_size*num_images)
    # Shuffle list
    images_info = images_info.sample(frac=1)
    for k in range(num_folds):
        images_info[:k*num_test][['TT']] = 'train'
        images_info[k*num_test:(k+1)*num_test][['TT']] = 'test'
        images_info[(k+1)*num_test:][['TT']] = 'train'
        file_name = images_info_csv.parent / (images_info_csv.stem + f'_{k}.csv')
        images_info.to_csv(file_name, index=False, header=False)
        print(f'Saving to: {file_name}')

#%%
def img_list_from_split(split_file, filterby):
    images_info = pd.read_csv(split_file, names=['Patient', 'Condition', 'TT'], dtype={'Patient': str, 'Condition': str, 'TT': str})
    return list(images_info[images_info['TT'] == filterby]['Patient'])


# This function gives a list of train and test directory, and optionally save the 
# train and test into a csv file
def divide_train_test(rootDir, trainRatio = 0.7, outDir = None):
    # Divide train and test set
    trainDirs = []
    testDirs = []
    
    diseases = os.listdir(rootDir)
    
    for d in diseases:
        diseaseDir = rootDir + d + "/"
        patientDirs = [ (diseaseDir + p + "/", p, d) for p in os.listdir(diseaseDir)]
        
        num_patients = len(patientDirs)
        idx = np.random.permutation(num_patients)
        
        trainDirs.extend([patientDirs[i] for i in idx[:int(trainRatio*num_patients)]])
        testDirs.extend([patientDirs[i] for i in idx[int(trainRatio*num_patients):]])
    
    # Write out the train test split
    if outDir is not None:
        with open(outDir, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for tr in trainDirs:
                writer.writerow(list(tr) + ['train'])
            for tr in testDirs:
                writer.writerow(list(tr) + ['test'])
    return trainDirs, testDirs
#%%
def create_train_test_split(inputdir, outDir, trainRatio = 0.8):
    """ Create random split from images in directory specified below
    """
    inputdir = Path(inputdir)
    img_list = [nii for nii in inputdir.iterdir() if nii.is_file() and not nii.name.endswith("_seg.nii.gz")]
    num_img = len(img_list)
    idx = np.random.permutation(num_img)
    train_img = [img_list[i].stem[:-4] for i in idx[:int(trainRatio*num_img)]]
    test_img  = [img_list[i].stem[:-4] for i in idx[int(trainRatio*num_img):]]
    with open(outDir, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for img in train_img:
            writer.writerow([img, 'NA', 'train'])
        for img in test_img:
            writer.writerow([img, 'NA', 'test'])

#%%
# This function read in the split file created by divide_train_test
def read_split(split_file, im_dir):
    """
    Returns tuple (for train, val, and test) of lists of length 3
        - image_path
        - patient ID
        - disease
    """
    trainDirs = []
    valDirs = []
    testDirs = []
    
    # split = pd.read_csv(c.train_test_csv, names=["patient", "condition", "train_test"])
    # split.insert(0, "path", c.data_nh_1mm)
    # split.drop(columns=["condition"], inplace=True)
    # split_train = split[split["train_test"]=="train"]
    # split_test = split[split["train_test"]=="test"]
    # trainDirs = split_train.to_numpy()
    # testDirs = split_test.to_numpy()
    with open(split_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader: #TODO: currently includes disease (:2)??
            if "train" in row:
                trainDirs.append([im_dir] + row[:2])
            elif "val" in row:
                valDirs.append([im_dir] + row[:2])
            elif "test" in row:
                testDirs.append([im_dir] + row[:2])
    return trainDirs, valDirs, testDirs

def create_split(csv_file, origin="UCSF"):
    df = pd.read_csv(csv_file, header="infer")
    df = df[df['Origin']==origin]
    return df


def create_3x_val(csv_file, origin):
    df = create_split(csv_file, origin)
    df1 = df.copy(deep=True)
    df2 = df.copy(deep=True)
    df3 = df.copy(deep=True)
    n = len(df)
    for i in range(n):
        if i%3 == 0:
            df1.iloc[i, df1.columns.get_loc('Origin')] = "train"
            df2.iloc[i, df2.columns.get_loc('Origin')] = "train"
            df3.iloc[i, df3.columns.get_loc('Origin')] = "test"
        elif i%3 == 1:
            df1.iloc[i, df1.columns.get_loc('Origin')] = "train"
            df2.iloc[i, df2.columns.get_loc('Origin')] = "test"
            df3.iloc[i, df3.columns.get_loc('Origin')] = "train"
        elif i%3 == 2:
            df1.iloc[i, df1.columns.get_loc('Origin')] = "test"
            df2.iloc[i, df2.columns.get_loc('Origin')] = "train"
            df3.iloc[i, df3.columns.get_loc('Origin')] = "train"
    print(df1.head())
    print(df2.head())
    print(df3.head())

    return (df1, df2, df3)

