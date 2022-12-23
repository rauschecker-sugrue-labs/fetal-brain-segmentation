from pathlib import Path

def diff_2lists(li1: list, li2: list) -> list:
    return list(list(set(li2)-set(li1)) + list(set(li1)-set(li2)))

def check_seg(data_dir: Path) -> {}:
    """ Check if the number of segmentation files matches with the number of images
        Return dictionary with list of missing segmentations and list of extra segmentations
    """
    image_names = [f.stem[:-4] for f in data_dir.iterdir() if f.name.endswith('.nii.gz') and 'seg' not in f.name]
    seg_names = [f.stem[:-8] for f in data_dir.iterdir() if f.name.endswith('_seg.nii.gz')]

    diff = diff_2lists(image_names, seg_names)
    res = {'missing seg': [], 'extra seg': []}
    for d in diff:
        if d in image_names:
            res['missing seg'].append(d)
        else:
            res['extra seg'].append(d)
    return res

def complete_images(data_dir: Path) -> []:
    """ Return list of images that have a segmentation
    """
    image_names = [f.stem[:-4] for f in data_dir.iterdir() if f.name.endswith('.nii.gz') and 'seg' not in f.name]
    missing_data = check_seg(data_dir)
    return diff_2lists(image_names, missing_data['missing seg'])