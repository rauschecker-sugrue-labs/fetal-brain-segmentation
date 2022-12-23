import os
from shutil import copy2, copyfile, copytree, rmtree
from tqdm.auto import tqdm
import pandas as pd

from utils.datasplit import create_split
from pathlib import Path


def move_file(filename, _from, _to):
    """
    Moves the file named filename from _from to _to
    Inputs:
        filename: name of the file
        _from: Path() object
        _to: Path() object
    """
    _to.mkdir(parents=True, exist_ok=True)
    if (_from.is_dir() and (_from/filename).is_file() and _to.is_dir()):
        copyfile(_from/filename, _to/filename)


def move_images_predict(_from, _to, csv_file):
    """ Moves images for prediction
    Args:
        _from:  input folder with raw/ and preprocessed/
        _to:    detination folder for the images (predict/experiment_name/)
    """
    # Get file names
    df = pd.read_csv(csv_file, names=['Patient', 'Condition', 'tt'])
    _from_data_raw = _from / "raw/"
    _from_data_nh = _from / "preprocessed/noheader/"
    _from_data_nh_1mm = _from / "preprocessed/noheader_1mm/"
    _to_data_raw = _to / "raw/"
    _to_data_nh = _to / "preprocessed/noheader/"
    _to_data_nh_1mm = _to / "preprocessed/noheader_1mm/"

    # Move files
    loop_patients = tqdm(df['Patient'], desc="Copying files", position=0)
    for patient in loop_patients:
        #raw
        move_file(str(patient)+".nii.gz", _from_data_raw, _to_data_raw)
        move_file(str(patient)+"_seg.nii.gz", _from_data_raw, _to_data_raw)
        #noheader
        move_file(str(patient)+".nii.gz", _from_data_nh, _to_data_nh)
        move_file(str(patient)+"_seg.nii.gz", _from_data_nh, _to_data_nh)
        #preprocessed 1mm
        move_file(str(patient)+"_1x1x1.nii.gz", _from_data_nh_1mm, _to_data_nh_1mm)
        move_file(str(patient)+"_seg_1x1x1.nii.gz", _from_data_nh_1mm, _to_data_nh_1mm)

def copy_model(model_name, root, new_name):
    model_dir = root / "models/"
    new_dir = model_dir / new_name / "model"
    if os.path.isdir(new_dir):
        rmtree(new_dir)
    return copytree(model_dir / model_name / "model", new_dir)

