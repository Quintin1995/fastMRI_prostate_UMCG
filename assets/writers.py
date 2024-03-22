import os
import SimpleITK as sitk
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd

from assets.util import dt_str
from assets.util import decipher


def write_pat_info_to_file(patients, logger, out_dir, key: str=None):
    """	
    Write patient info to a csv file.
    """
    
    assert key is not None, "key must be provided to decipher the anon_id"
    
    df = pd.DataFrame(columns=['seq_id', 'id', 'anon_id', 'data_dir'])
    new_rows = []

    for seq_id, anon_id, pat_dir in patients:
        id = decipher(anon_id, key=key)
        
        new_row = pd.DataFrame({
            'seq_id': [seq_id],
            'id': [id],
            'anon_id': [anon_id],
            'data_dir': [pat_dir]
        })
        new_rows.append(new_row)
        
    df = pd.concat([df] + new_rows, ignore_index=True)
    
    df['seq_id'] = "'" + df['seq_id'].astype(str)
    df['id'] = "'" + df['id'].astype(str)
    df['anon_id'] = "'" + df['anon_id'].astype(str)
    df['data_dir'] = "'" + df['data_dir'].astype(str)
    
    dirname = out_dir / 'mappings'
    df.to_csv(dirname / 'patient_info.csv', index=False, sep=';')
    logger.info(f"Saved patient info to {dirname / 'patient_info.csv'}")


def save_numpy_rss_as_nifti(image: np.ndarray, fname: str, dir: str, logger: logging.Logger = None) -> None:
    """
    Description:
        Save the given numpy array as a nifti file.
    Args:
        image (np.ndarray): The image to save.
        fname (str): The file name.
    Returns:
        None
    """

    assert image.ndim == 3, "image should have 3 dimensions: (n_slices, n_freq, n_phase)"

    fpath = os.path.join(dir, f"{fname}_{dt_str()}.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(image), fpath)
    logger.info(f"\tSaved nifti image to {fpath}")
    
    
def save_crop_to_file(dir: str, pat_num: str, kspace, cropsize=10, avg=0, slice=0, coil=0) -> None:
    '''
    Arguments:
        - dir: directory to save the crop to
        - pat_num: patient number
        - kspace: numpy array of kspace data in shape (navgs, nslices, ncoils, rNx, eNy + 1) complex
        - cropsize: size of the crop
        - avg: average to take the crop from
        - slice: slice to take the crop from
        - coil: coil to take the crop from
    '''
    
    # build a suitable filename that includes the avg, slice and coil and crop
    fname = os.path.join(dir, f'{pat_num}_crop_avg{avg}_slice{slice}_coil{coil}.png')

    plt.imsave(fname, np.abs(kspace[avg, slice, coil, 0:cropsize, 0:cropsize]), cmap='gray')
    print(f"saved crop to {fname}")


def save_crops_to_file(dir: str, pat_num: str, kspace, cropsize=10, slice=0, coil=0) -> None:
    '''
    Arguments:
        - dir: directory to save the crop to
        - pat_num: patient number
        - kspace: numpy array of kspace data in shape (navgs, nslices, ncoils, rNx, eNy + 1) complex
        - cropsize: size of the crop
        - slice: slice to take the crop from
        - coil: coil to take the crop from
    '''
    
    # build a suitable filename that includes the avg, slice and coil and crop
    fname = os.path.join(dir, f'{pat_num}_crop_avg012_slice{slice}_coil{coil}.png')

    crop1 = np.abs(kspace[0, slice, coil, 0:cropsize, 0:cropsize])
    crop2 = np.abs(kspace[1, slice, coil, 0:cropsize, 0:cropsize])
    crop3 = np.abs(kspace[2, slice, coil, 0:cropsize, 0:cropsize])

    #combine them row wise for visualization
    combined = np.concatenate((crop1, crop2, crop3), axis=1)

    plt.imsave(fname, combined, cmap='gray')
    print(f"saved crop to {fname}")


def save_np_array(nparray: np.ndarray, fname: str)-> None:
    '''
    Arguments:
        - nparray: numpy array to save
        - fname: filename to save to
    '''
    np.save(fname, nparray)
    print(f"save to {nparray}")