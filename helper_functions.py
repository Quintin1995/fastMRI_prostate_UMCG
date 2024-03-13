import os
import numpy as np
import h5py
import ismrmrd
import ismrmrd.xsd
import glob
import numpy as np
import SimpleITK as sitk
from ismrmrdtools import transform
import matplotlib.pyplot as plt
import h5py
import logging
import fnmatch
from typing import List, Dict, Tuple
import argparse
import sqlite3

from xml.dom import minidom
import xml.etree.ElementTree as ET
import pydicom
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from pathlib import Path
import subprocess
from datetime import datetime
from cipher import encipher, decipher

KEY = '7531598426'


def safe_kspace_slice_to_png(kspace: np.ndarray, dir: str, titlepart="", do_log=True, slice_no=0, coil_no=0, avg_to_add=(0,1)) -> None:
    '''
    Description:
        Visualize the k-space of the given slice and coil.
        And saves the figure to the given directory.
    Args:
        kspace (np.ndarray): The k-space data.
        dir (str): The output directory.
        titlepart (str): The title part. Used for the file name and title.
        do_log (bool): Whether to take the log of the k-space.
        slice_no (int): The slice number.
        coil_no (int): The coil number.
        avg_to_add (tuple): The averages to add together.
    Returns:
        None
    '''
    assert kspace.ndim == 5, "image should have 5 dimensions: (n_avg, n_slices, n_coils, n_freq, n_phase)"

    # Add the first and second average together in a new array so that the average dimension is removed
    kspace = np.add(kspace[avg_to_add[0],...], kspace[avg_to_add[1],...])

    # Get the slice and coil image
    kspace = kspace[slice_no, coil_no, :, :]

    
    if do_log:
        kspace = np.log(np.abs(kspace) + 1e-6)  # 1e-6 is added to avoid log(0)

    # compute the location of the center where the kspace is max value
    cen_x, cen_y = np.unravel_index(np.argmax(kspace), kspace.shape)
    print(f"Computed center indexes: {cen_x, cen_y}\nThis is an approximation of the center.")

    plt.figure(figsize=(10,10), dpi=300)
    plt.imshow(np.real(kspace), cmap='gray', interpolation='none')
    plt.colorbar(label='Magnitude')
    if do_log:
        plt.title(f'Logarithmic K-space slice 0 coil 0 - {titlepart}')
        fname = f'{dir}/ksp_slice0_coil0_log_{titlepart}.png'
    else:
        plt.title(f'K-space slice 0 coil 0 - {titlepart}')
        fname = f'{dir}/ksp_slice0_coil0_{titlepart}.png'

    # add figure text of center
    plt.figtext(0.5, 0.02, f"Center indexes: {cen_x, cen_y}", wrap=True, horizontalalignment='center', fontsize=10)
    plt.xlabel('Phase Encoding')
    plt.ylabel('Frequency Encoding')
    plt.savefig(fname)
    plt.close()
    print(f'saved k-space visualization for slice 0 and coil 0 to: {fname}')


# create a function out of the code below
def rss_recon_from_ksp(ksp: np.array, averages=(0,1), verbose=False, printkey="", logger: logging.Logger = None) -> np.ndarray:
    """
    Perform root sum of squares reconstruction on the given k-space.
    
    Parameters:
        ksp (np.array): The k-space data.
        averages (tuple): The averages to add together.
        verbose (bool): Whether to print information.
        printkey (str): The print key.
    Returns:
        np.array: The reconstructed image.
    """
    # assert ndim equal to 5
    assert ksp.ndim == 5, "ksp should have 5 dimensions: (n_avg, n_slices, n_coils, n_freq, n_phase)"

    logger.info(f"\tRSS reconstruction from kspace")

    n_avg, n_slices, n_coils, n_freq, n_phase = ksp.shape

    # Add the first and second average together in a new array so that the average dimension is removed
    ksp   = np.add(ksp[averages[0],...], ksp[averages[1],...])
    image = np.zeros((n_slices, n_freq, n_phase), dtype=np.float32)

    if verbose:
        logger.info(f"{printkey} - kspace shape: ({n_avg, n_slices, n_coils, n_freq, n_phase})")
        logger.info(f'{printkey} - Image  shape: {image.shape}')
        logger.info(f"{printkey} - kspace shape collapsed from averages: {ksp.shape}")

    for slice_ in range(n_slices):

        ksp_slice_coils = ksp[slice_, :, :, :]

        # Perform inverse Fourier transform to obtain image
        slice_im_coils = transform.transform_kspace_to_image(ksp_slice_coils, [1, 2])

        # Root sum of squares on slice
        image[slice_, :, :] = np.sqrt(np.sum(np.abs(slice_im_coils) ** 2, 0))

    return image


def dt_str():
    now = datetime.now()
    return now.strftime("%m%d%H%M")


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


def create_dir(directory_path, verbose=True):
    '''
    Create a directory if it does not exist
    Arguments:
        - directory_path: path to the directory to create
        - verbose: whether to print the directory created
    '''
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        if verbose:
            print(f"Directory '{directory_path}' created successfully.")
    else:
        if verbose:
            print(f"Directory '{directory_path}' already exists.")


def visualize_dicom_slice_acquisition(dcm_dir_path: str, dir='home1/p290820/tmp/') -> None:
    """
    Creates a scatter plot visualizing the Acquisition Time versus Instance Number (slice location)
    for DICOM files in the specified directory. Saves the plot to /home1/p290820/tmp/somefile.png.
    
    Parameters:
        dcm_dir_path (str): Path to the directory containing the DICOM files.
    
    Returns:
        None
    """
    # Initialize lists to store Instance Numbers and Acquisition Times
    instance_numbers = []
    acquisition_times = []

    # Loop through all DICOM files in the directory
    for filename in os.listdir(dcm_dir_path):
        filepath = os.path.join(dcm_dir_path, filename)
        
        # Read DICOM file
        dicom_file = pydicom.dcmread(filepath)
        
        # Extract Instance Number and Acquisition Time
        instance_number = int(dicom_file.InstanceNumber)
        acquisition_time = float(dicom_file.AcquisitionTime)
        
        # Append to lists
        instance_numbers.append(instance_number)
        acquisition_times.append(acquisition_time)

    # Sort by Instance Number for plotting
    sorted_indices = sorted(range(len(instance_numbers)), key=lambda k: instance_numbers[k])
    sorted_acquisition_times = [acquisition_times[i] for i in sorted_indices]
    sorted_instance_numbers = [instance_numbers[i] for i in sorted_indices]

    # print both list in the correct order
    print(sorted_instance_numbers)
    print(sorted_acquisition_times)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_instance_numbers, sorted_acquisition_times)
    plt.title('Acquisition Time vs Slice Number')
    plt.xlabel('Instance Number (Slice Location)')
    plt.ylabel('Acquisition Time')
    plt.grid(True)
    fpath = Path(dir, "_sliceNum_vs_sliceTime.png")
    plt.savefig('fpath')
    print(f"Plot saved to {fpath}")


def safe_rss_to_nifti_file(
    kspace: np.ndarray,
    fname_part = '',
    do_round=True,
    dir: Path = Path('/home1/p290820/tmp'),
    logger: logging.Logger = None
) -> None:
    """
    Description:
        Perform root sum of squares reconstruction on the given k-space.
    Args:
        kspace (np.ndarray): The k-space data. 5D array with dimensions [num_averages, num_slices, num_coils, num_readout_points, num_phase_encode_steps]
    Returns:
        None
    """

    assert kspace.ndim == 5, "image should have 5 dimensions: (n_avg, n_slices, n_coils, n_freq, n_phase)"

    rss_recon = rss_recon_from_ksp(kspace, averages=(0,1), printkey="safe_rss_to_nifti_file_reorder", logger=logger)

    if do_round:
        rss_recon = np.round(rss_recon*1000, decimals=3)

    save_numpy_rss_as_nifti(rss_recon, fname=f"{fname_part}_rss_recon", dir=dir, logger=logger)


def echo_train_length(dset) -> int:
    '''
    Description: 
        This function determines the echo train length from the .mrd file
    Assumptions:
        - The noise acquisitions are made in the beginning.
        - The noise acquisition is made in the same slice as the first ET.
        - There are multiple slices.
        - After the first ET, it moves to the first ET of a different slice.
        - The first ET has the same length as the other ones.
    Arguments:
        - dset: ismrmrd.Dataset object
    Returns:
        - etl: echo train length
    '''
    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n).isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            firstacq = n
            break
    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n)._head.idx.slice != dset.read_acquisition(0)._head.idx.slice:
            return n - firstacq
    raise Exception("Couldn't find different slices in the dataset")


def echo_train_count(dset, echo_train_len=25) -> int:
    '''
    Description:
        This function determines the echo train count from the .mrd file
    Assumptions:
        - the assumptions of echo_train_length
        - There are at least 2 averages (idx 0 and 1) avg0 and avg1
        - Higher index averages are acquired later
    Arguments:
        - dset: ismrmrd.Dataset object
        - echo_train_len: echo train length
    Returns:
        - echo_train_count: echo train count
    '''
    enc = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header()).encoding[0]

    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        raise Exception("Couldn't find different slices in the dataset")
    
    count = 0
    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n)._head.idx.average == 2:
            break
        if dset.read_acquisition(n)._head.idx.average == 1:
            count += 1

    return int(count/(nslices * echo_train_len))


def get_num_slices(dset):
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]
    if enc.encodingLimits.slice is not None:
        return enc.encodingLimits.slice.maximum + 1
    else:
        raise Exception("Couldn't find different slices in the dataset")


def get_num_averages(dset, firstacq: int):
    averages = 0
    for acqnum in range(firstacq, dset.number_of_acquisitions()):
        averages = dset.read_acquisition(acqnum)._head.idx.average
    return averages + 1


def case_insensitive_glob(pattern) -> List[str]:
    '''
    Arguments:
        - pattern: pattern to match
    Returns:
        - matches: list of matches
    '''
    matches = []
    for file in glob.glob(pattern):
        if fnmatch.fnmatch(file.lower(), pattern.lower()):
            matches.append(file)
    return matches


def get_patient_id(path_mrd, pos=12) -> str:
    '''
    Arguments:
        - path: path to the .mrd file 
        - pos: position of the patient id in the path string
    Returns:
        - pat_num: patient number as a string
    '''
    try:
        pat_num = path_mrd.split('/')[pos].split('_')[0]
        print(f"got patient id from path_mrd {path_mrd}")
        print(f"pat_num: {pat_num}")
    except:
        try:
            pat_num = path_mrd.split('\\')[10].split('_')[0]
            print(f"got patient id from path_mrd {path_mrd}")
            print(f"pat_num: {pat_num}")
        except:
            print(f"could not get patient id from path_mrd {path_mrd}")

    # check if patient id is of length 4
    if len(pat_num) != 4:
        print(f"pat_num {pat_num} is not of length 4")

    return pat_num


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


# create a function that saves three crops combined into one image
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


def view_stats_kspace(kspace: np.ndarray, tmpdir: str, pat_num: str, save_crops: False) -> None:
    '''
    Arguments:
        - kspace: numpy array of kspace data in shape (navgs, nslices, ncoils, rNx, eNy + 1) complex
        - tmpdir: directory to save the crops to
        - pat_num: patient number
        - save_crops: whether to save crops of the first two slices
    '''
    print(f"kspace shape: {kspace.shape}")
    print(f"kspace dtype: {kspace.dtype}")

    if save_crops:
        save_crop_to_file(tmpdir, pat_num, kspace, cropsize=30, avg=0, slice=0, coil=0)
        save_crop_to_file(tmpdir, pat_num, kspace, cropsize=30, avg=1, slice=0, coil=0)
        save_crop_to_file(tmpdir, pat_num, kspace, cropsize=30, avg=2, slice=0, coil=0)
        save_crops_to_file(tmpdir, pat_num, kspace, cropsize=30, slice=0, coil=0)


def save_np_array(nparray: np.ndarray, fname: str)-> None:
    '''
    Arguments:
        - nparray: numpy array to save
        - fname: filename to save to
    '''
    np.save(fname, nparray)
    print(f"save to {nparray}")


def calculate_zero_padding_PE(kspace: np.ndarray, slice_no: int = 0, coil_no: int = 0):
    """
    Description:
    ------------
    Calculate the amount of zero padding in the k-space data in the phase encoding direction.
    This is done by summing the real part of the average k-space data over the averages and coils.
    The resulting 2D array is summed over the frequency encoding direction.
    The amount of zeros in this array is the amount of zero padding.

    Parameters:
    -----------
    kspace: np.ndarray
        The k-space data of shape (n_avg, n_slice, n_coil, n_freq, n_ph)
    slice_no: int
        The slice number to calculate the zero padding for
    coil_no: int
        The coil number to calculate the zero padding for
    Returns:
    --------
    zero_padding: int
        The amount of zero padding in the k-space data
    """
    avg_kspace = kspace[:, slice_no, coil_no, :, :]
    kspace_2d = np.sum(np.real(avg_kspace), axis=0)
    
    # perform the same computation but faster
    zero_padding = np.sum(np.sum(kspace_2d, axis=0) == 0)

    # also calculate the phase encoding line index of the zero paddings
    zero_padding_indices = np.where(np.sum(kspace_2d, axis=0) == 0)[0]

    # the first phase line is zeros but not considered zero padding
    if zero_padding_indices[0] == 0:
        zero_padding_indices = zero_padding_indices[1:]
        zero_padding -= 1

    return zero_padding, zero_padding_indices


def convert_ismrmrd_headers_to_dict(ismrmrd_headers):
    ns = "{http://www.ismrm.org/ISMRMRD}"

    new_hdrs = {
        f"{ns}ismrmrdHeader": {
            f"{ns}studyInformation": {
                f"{ns}studyTime": str(ismrmrd_headers.measurementInformation.seriesTime),
            },
            f"{ns}measurementInformation": {
                f"{ns}measurementID": str(ismrmrd_headers.measurementInformation.measurementID), 
                f"{ns}patientPosition": str(ismrmrd_headers.measurementInformation.patientPosition),
                f"{ns}protocolName": str(ismrmrd_headers.measurementInformation.protocolName),
                f"{ns}measurementDependency": {
                    f"{ns}dependencyType": str(ismrmrd_headers.measurementInformation.measurementDependency[0].dependencyType),
                    f"{ns}measurementID": str(ismrmrd_headers.measurementInformation.measurementDependency[0].measurementID),
                },
                f"{ns}measurementDependencyType": {
                    f"{ns}dependencyType": str(ismrmrd_headers.measurementInformation.measurementDependency[1].dependencyType),
                    f"{ns}measurementID": str(ismrmrd_headers.measurementInformation.measurementDependency[1].measurementID),
                },
                f"{ns}frameOfReferenceUID": str(ismrmrd_headers.measurementInformation.frameOfReferenceUID),
            },
            f"{ns}acquisitionSystemInformation": {
                f"{ns}systemVendor": str(ismrmrd_headers.acquisitionSystemInformation.systemVendor),
                f"{ns}systemModel": str(ismrmrd_headers.acquisitionSystemInformation.systemModel),
                f"{ns}systemFieldStrength_T": str(ismrmrd_headers.acquisitionSystemInformation.systemFieldStrength_T),
                f"{ns}relativeReceiverNoiseBandwidth": str(ismrmrd_headers.acquisitionSystemInformation.relativeReceiverNoiseBandwidth),
                f"{ns}receiverChannels": str(ismrmrd_headers.acquisitionSystemInformation.receiverChannels),
                **{
                    f"{ns}coilLabel{i}": {
                        f"{ns}coilNumber": str(ismrmrd_headers.acquisitionSystemInformation.coilLabel[i].coilNumber),
                        f"{ns}coilName": str(ismrmrd_headers.acquisitionSystemInformation.coilLabel[i].coilName),
                    }
                    for i in range(len(ismrmrd_headers.acquisitionSystemInformation.coilLabel))
                },
                f"{ns}institutionName": str(ismrmrd_headers.acquisitionSystemInformation.institutionName),
                f"{ns}deviceID": str(ismrmrd_headers.acquisitionSystemInformation.deviceID),
            },
            f"{ns}experimentalConditions":{
                f"{ns}H1resonanceFrequency_Hz": str(ismrmrd_headers.experimentalConditions.H1resonanceFrequency_Hz),
            },
            f"{ns}encoding": {
                f"{ns}encodedSpace": {
                    f"{ns}matrixSize": {
                        f"{ns}x": str(ismrmrd_headers.encoding[0].encodedSpace.matrixSize.x),
                        f"{ns}y": str(ismrmrd_headers.encoding[0].encodedSpace.matrixSize.y),
                        f"{ns}z": str(ismrmrd_headers.encoding[0].encodedSpace.matrixSize.z),
                    },
                    f"{ns}fieldOfView_mm": {
                        f"{ns}x": str(ismrmrd_headers.encoding[0].encodedSpace.fieldOfView_mm.x),
                        f"{ns}y": str(ismrmrd_headers.encoding[0].encodedSpace.fieldOfView_mm.y),
                        f"{ns}z": str(ismrmrd_headers.encoding[0].encodedSpace.fieldOfView_mm.z),
                    }
                },
                f"{ns}reconSpace": {
                    f"{ns}matrixSize": {
                        f"{ns}x": str(ismrmrd_headers.encoding[0].reconSpace.matrixSize.x),
                        f"{ns}y": str(ismrmrd_headers.encoding[0].reconSpace.matrixSize.y),
                        f"{ns}z": str(ismrmrd_headers.encoding[0].reconSpace.matrixSize.z),
                    }
                },
                f"{ns}encodingLimits": {
                    f"{ns}kspace_encoding_step_1": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.kspace_encoding_step_1.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.kspace_encoding_step_1.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.kspace_encoding_step_1.center),
                    },
                    f"{ns}kspace_encoding_step_2": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.kspace_encoding_step_2.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.kspace_encoding_step_2.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.kspace_encoding_step_2.center),
                    },
                    f"{ns}average": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.average.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.average.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.average.center),
                    },
                    f"{ns}slice": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.slice.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.slice.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.slice.center),
                    },
                    f"{ns}contrast": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.contrast.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.contrast.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.contrast.center),
                    },
                    f"{ns}phase": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.phase.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.phase.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.phase.center),
                    },
                    f"{ns}repetition": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.repetition.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.repetition.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.repetition.center),
                    },
                    f"{ns}set": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.set.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.set.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.set.center),
                    },
                    f"{ns}segment": {
                        f"{ns}minimum": str(ismrmrd_headers.encoding[0].encodingLimits.segment.minimum),
                        f"{ns}maximum": str(ismrmrd_headers.encoding[0].encodingLimits.segment.maximum),
                        f"{ns}center": str(ismrmrd_headers.encoding[0].encodingLimits.segment.center),
                    },
                },
                f"{ns}trajectory": str(ismrmrd_headers.encoding[0].trajectory.value),
                f"{ns}parallelImaging": {
                    f"{ns}accelerationFactor": {
                        f"{ns}kspace_encoding_step_1": str(ismrmrd_headers.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1),
                        f"{ns}kspace_encoding_step_2": str(ismrmrd_headers.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2),
                    },
                    f"{ns}calibrationMode": str(ismrmrd_headers.encoding[0].parallelImaging.calibrationMode.value),
                    f"{ns}interleavingDimension": str(ismrmrd_headers.encoding[0].parallelImaging.interleavingDimension.value),
                }
            },
            f"{ns}sequenceParameters": {
                f"{ns}TR": str(ismrmrd_headers.sequenceParameters.TR[0]),
                f"{ns}TE": str(ismrmrd_headers.sequenceParameters.TE[0]),
                f"{ns}TI": str(ismrmrd_headers.sequenceParameters.TI[0]),
                f"{ns}sequence_type": str(ismrmrd_headers.sequenceParameters.sequence_type),
                f"{ns}echo_spacing": str(ismrmrd_headers.sequenceParameters.echo_spacing[0]),
            },
            f"{ns}userParameters": {
                f"{ns}userParameterLong": {
                    f"{ns}name": str(ismrmrd_headers.userParameters.userParameterLong[64].name),
                    f"{ns}value": str(ismrmrd_headers.userParameters.userParameterLong[64].value),
                }
            }
        }
    }
        
    return new_hdrs


def has_key_in_h5(h5_fpath: str, key: str, logger: logging.Logger) -> bool:
    """
    Verify the validity of an h5 file by checking the presence of a key.

    Args:
        h5_fpath (str): Path to the h5 file.
        key (str): Key to check for.

    Returns:
        bool: True if the key is found, False otherwise.
    """
    with h5py.File(h5_fpath, 'r') as h5_file:
        if key not in h5_file.keys():
            logger.info(f"\tThe '{key}' key was NOT FOUND in the h5 file.")
            return False

        logger.info(f"\tThe '{key}' key was FOUND.")
        return True

        
def has_correct_shape(h5_fpath: str, logger: logging.Logger) -> bool:
    """
    Verify the validity of an h5 file by checking the shape of the 'kspace' key.

    Args:
        h5_fpath (str): Path to the h5 file.

    Returns:
        bool: True if the shape of the 'kspace' key is correct, False otherwise.
    """
    with h5py.File(h5_fpath, 'r') as h5_file:
        if h5_file["kspace"].ndim != 5:
            logger.info(f"\tThe 'kspace' key has an incorrect number of dimensions. Shape: {h5_file['kspace'].shape}")
            return False

        logger.info(f"\tThe 'kspace' key has the correct number of dimensions. Shape: {h5_file['kspace'].shape}")
        return True
    

def encipher(patient_id: str, key: str):
    '''
    Description:
        Enciphers the patient id to the anonymized patient id
    Arguments:
        - patient_id: patient id
        - key: key to use for enciphering
    Returns:
        - anon_pat_id: anonymized patient id
    '''
    output = []
    non_digit_char_count = 0

    for c_idx, char in enumerate(patient_id):
        if not str.isdigit(char):
            output.append(char)
            non_digit_char_count += 1
            continue
        key_idx = c_idx - non_digit_char_count
        k = int(key[key_idx])
        out_char = (int(char) + k) % 10
        output.append(str(out_char))
    
    return f"ANON{''.join(output)}"


def decipher(anon_id: str, key: str):
    '''
    Description:
        Deciphers the anonymized patient id to the unanonymized patient id
    Arguments:
        - anon_id: anonymized patient id
        - key: key to use for deciphering
    Returns:
        - unanon_pat_id: unanonymized patient id
    '''

    output = []
    non_digit_char_count = 0

    for c_idx, char in enumerate(anon_id[4:]):
        if not str.isdigit(char):
            output.append(char)
            non_digit_char_count += 1
            continue
        key_idx = c_idx - non_digit_char_count
        k = int(key[key_idx])
        out_char = (int(char) - k) % 10
        output.append(str(out_char))
    
    return ''.join(output)


def get_mapping_patient_ids(workdir, key: str, verbose=False) -> List[Dict[str, str]]:
    '''
    Description:
        Gets the mapping patient ids from the path and ciphers them to get the anonID that is used for the dicom files
        This way they can be linked with the kspace (mrd) files
    Arguments:
        - workdir: working directory
        - verbose: whether to print the patient ids
    Returns:
        - pat_dicts: list of dicts that contains the patient id, the anonymized patient id, the unanonymized patient id and the dirname
    '''

    # create a list of dicts that contains the patient id, the anonymized patient id, the unanonymized patient id and the dirname
    pat_dicts = []

    found_paths = glob.glob(os.path.join(workdir, "input", "export_scanner", "*", "dicoms", "*"))

    if verbose:
        print(f"found {len(found_paths)} dicom dirs")

    for path in found_paths:
        try:
            pat_id        = path.split('/')[-1].split('_')[-1].strip()
            anon_pat_id   = encipher(pat_id, key=key)
            unanon_pat_id = decipher(anon_pat_id, key=key)
            patnum_str    = get_patient_id(path)

            assert pat_id == unanon_pat_id, f"pat_id {pat_id} does not match unanon_pat_id {unanon_pat_id}"
            assert len(pat_id) == 7, f"pat_id {pat_id} is not of length 7"
            assert len(anon_pat_id) == 11, f"anon_pat_id {anon_pat_id} is not of length 11"
            assert len(unanon_pat_id) == 7, f"unanon_pat_id {unanon_pat_id} is not of length 7"
            assert len(patnum_str) == 4, f"patnum_str {patnum_str} is not of length 4"

            pat_dicts.append({
                "pat_id": pat_id,
                "anon_pat_id": anon_pat_id,
                "unanon_pat_id": unanon_pat_id,
                "pat_str": patnum_str,
                "dirname": f"{patnum_str}_patient_umcg_done",
                "path": path,
            })
        except Exception as e:
            print(f"Patient ids could not be extracted from path {path} and dicom dir")
            print(e)
    
    if verbose:
        print(f"found {len(pat_dicts)} patient ids")
        for pat_dict in pat_dicts:
            for key, value in pat_dict.items():
                print(f"\t{key}: {value}")
            print("")

    return pat_dicts


def print_element(element, depth=0) -> None:
    '''
    Description:
        - print an XML element and its depth in the tree
    Arguments:
        - element: the XML element
        - depth: the current depth in the tree (default is 0 for the root)
    '''
    print('\t' * depth + f"{element.tag}: {element.text}")
    for child in element:
        print_element(child, depth + 1)


def get_headers_from_h5(fpath: str, verbose=False):
    '''
    Description:
        - get the headers from a .mrd file
    Arguments:
        - fpath: path to the .mrd file
    Returns:
        - headers: dictionary with the headers
    '''

    with h5py.File(fpath, 'r') as hf:
        # Get the data from the ismrmrd_header dataset as bytes
        header_bytes = hf["ismrmrd_header"][()]

        # Decode the bytes to a string
        header_string = header_bytes.decode('utf-8')

        # Parse the XML string into an ElementTree object
        header_tree = ET.ElementTree(ET.fromstring(header_string))

        # Access the root element of the XML tree
        root = header_tree.getroot()

        # Now you can work with the XML object (root) and extract information as needed
        # For example, printing the tag and text of each element
        if verbose: 
            print_element(root)
        
        return root


def print_dict_values(d, depth=0):
    '''
    Print the values of a dictionary with a given depth
    '''
    for key, value in d.items():
        if isinstance(value, dict):
            print_dict_values(value, depth + 1)
        else:
            print(f"\t" * depth + f"{key}: {value}")


def get_headers_from_ismrmrd(fpath: str, verbose=False) -> dict:
    '''
    Description:
        - get the headers from a .mrd file
    Arguments:
        - fpath: path to the .mrd file
    Returns:
        - headers: dictionary with the headers
    '''
    dset = ismrmrd.Dataset(fpath, 'dataset', create_if_needed=False)
    headers = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

    if verbose:
        print("\tHEADERS found in .mrd file")
        print(f"\t\tHEADERS.SUBJECTINFORMATION")
        print_headers(headers.subjectInformation)
        print(f"\t\tHEADERS.MEASUREMENTINFORMATION")
        print_headers(headers.measurementInformation)
        print(f"\t\tHEADERS.ACQUISITIONSYSTEMINFORMATION")
        print_headers(headers.acquisitionSystemInformation)
        print(f"\t\tHEADERS.SEQUENCEPARAMETERS")
        print_headers(headers.sequenceParameters)
        print(f"\t\tHEADERS.USERPARAMETERS")
        print_headers(headers.userParameters)
        print(f"\t\tHEADERS.EXPERIMENTALCONDITIONS")
        print_headers(headers.experimentalConditions)
        print(f"\t\tHEADERS.ENCODEDSPACE")
        print_headers(headers.encoding[0].encodedSpace)
        print(f"\t\tHEADERS.TRAJECTORY")
        print_headers(headers.encoding[0].trajectory)
        print(f"\t\tHEADERS.RECONSPACE")
        print_headers(headers.encoding[0].reconSpace)
        print(f"\t\tHEADERS.ENCODINGLIMITS")
        print_headers(headers.encoding[0].encodingLimits)
        print(f"\t\tHEADERS.PARALLELIMAGING")
        print_headers(headers.encoding[0].parallelImaging)
        print(type(headers))

    return headers


def print_headers(header_key):
    """
    Print the header information of a given header key. 
    """
    header_dict = vars(header_key)
    for key, value in header_dict.items():
        print(f"\t\t\t{key}:", end="")
        if isinstance(value, type):
            for attr_name, attr_value in vars(value).items():
                print(f": {attr_name}: {attr_value}")
        else:
            print(f"\t\t\t{value}")


def convert_dict_to_xml(input_dict, root_element):
    """
    Converts a dictionary to an XML ElementTree.
    """
    def add_elements(d, parent_element):
        for k, v in d.items():
            if isinstance(v, dict):
                child_element = ET.SubElement(parent_element, k)
                add_elements(v, child_element)
            else:
                child_element = ET.SubElement(parent_element, k)
                child_element.text = v

    add_elements(input_dict, root_element)

    return root_element


def pretty_print_xml(xml_element):
    """
    Returns a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(xml_element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def encode_umcg_header_to_bytes(umcg_to_nyu_dict: dict) -> bytes:
    """
    Description:
        - Encode the UMCG headers into NYU headers and save it into a .h5 file.
    Arguments:
        - umcg_to_nyu_dict: mapping dictionary from UMCG headers to NYU headers
        - fpath: path to the .h5 file to save the headers
    """

    # Convert the dictionary to an XML object
    root_element = ET.Element("{http://www.ismrm.org/ISMRMRD}ismrmrdHeader")
    xml_element = convert_dict_to_xml(umcg_to_nyu_dict["{http://www.ismrm.org/ISMRMRD}ismrmrdHeader"], root_element)

    # Convert the XML object to a pretty-printed XML string
    xml_string = pretty_print_xml(xml_element)

    # Encode the XML string to bytes
    header_bytes = xml_string.encode('utf-8')

    return header_bytes


def analyze_kspace_3d_vol(ksp: np.ndarray, label="") -> None:
    '''
    Description:
        - Analyze the 3D kspace volume and print the statistics of the real and imaginary parts
    Arguments:
        - ksp: kspace data
        - label: label to print
    '''

    assert ksp.ndim == 5, f"kspace data is not 5D. Shape: {ksp.shape}"
    # assert complex data
    assert np.iscomplexobj(ksp), f"kspace data is not complex. Shape: {ksp.shape} and type {ksp.dtype}"

    # Compute the added 3D k-space volume once
    kspace_vol = np.add(ksp[0,:,0,:,:], ksp[1,:,0,:,:])
    
    # Compute real and imaginary parts
    kspace_vol_real = np.real(kspace_vol)
    kspace_vol_imag = np.imag(kspace_vol)
    
    # Print the statistics
    print(f"\t{label}")
    print(f"\t3d vol {label} = {kspace_vol.shape}")
    print(f"\treal min = {np.min(kspace_vol_real)}")
    print(f"\treal max = {np.max(kspace_vol_real)}")
    print(f"\treal median = {np.median(kspace_vol_real)}")
    print(f"\treal std = {np.std(kspace_vol_real)}")
    print(f"\timag min = {np.min(kspace_vol_imag)}")
    print(f"\timag max = {np.max(kspace_vol_imag)}")
    print(f"\timag median = {np.median(kspace_vol_imag)}")
    print(f"\timag std = {np.std(kspace_vol_imag)}")
    
    # Free up memory
    del kspace_vol, kspace_vol_real, kspace_vol_imag


def visualize_kspace_averages(kspace: np.ndarray, slice_idx: int, coil_idx: int, outdir: str, fname_id: str, figtext="") -> None:
    """
    Description:
        Plot the average lines of the given slice and coil.
        The average lines are colored by the average index.
        The average index is the index of the average in the k-space.
    Args:
        kspace (np.array): The k-space data.
        slice_idx (int): The slice index.
        coil_idx (int): The coil index.
        outdir (str): The output directory.
        fname_id (str): The file name identifier.
        figtext (str): The figure text.
    Returns:
        None
    """
    
    # Get the average measurements for the given slice and coil
    averages = kspace[:, slice_idx, coil_idx, :, :]
    n_avg, n_freq, n_ph = averages.shape
    
    # Create an array of zeros to color the lines
    color_array = np.zeros(averages[0].shape, dtype=np.uint8)

    # Loop over the phase encoding lines and average indices
    for ph_idx in range(n_ph):
        for avg_idx in range(n_avg):

            # Get the real part of the average
            line = np.real(averages[avg_idx, :, ph_idx])

            # If the line is not zero, color it with the average index + 1 (0 is zero-padding) 
            if np.sum(line) != 0:
                if np.sum(color_array[:, ph_idx]) != 0:
                    color_array[0:n_freq//2, ph_idx] = avg_idx + 1
                else:
                    color_array[:, ph_idx] = avg_idx + 1

    figtext += f"\nNumber of averages: {n_avg}. K-space shape: {kspace.shape} (copmlex)."
    figtext += "\nAvg 1 and 3 measure the same lines, creating a half-half color line. Real parts of k-space shown."

    fname = f'{outdir}/averages_vis_{fname_id}_s{slice_idx}c{coil_idx}.png'
    
    plt.figure(figsize=(10,10), dpi=300)
    plt.imshow(color_array, cmap='viridis', interpolation='none')
    plt.colorbar(ticks=range(n_avg+1), label='Average Number (0 is zero-padding)')
    plt.title(f'K-space averages for Slice {slice_idx} and Coil {coil_idx} - {fname_id}')
    plt.xlabel('Phase Encoding')
    plt.ylabel('Frequency Encoding')
    plt.figtext(0.5, 0.02, figtext, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig(fname)
    print(f'Saved average visualization for slice {slice_idx} and coil {coil_idx} to {fname}')
    plt.close()


def hist_real_and_imag(ksp: np.ndarray, title: str, slice_idx=0, coil_idx=0) -> None:
    '''
    Description:
        - Plot the histogram of the real and imaginary parts of the given slice and coil.
    Arguments:
        - ksp: kspace data (complex) numpy array
        - title: title of the plot
        - slice_idx: slice index
        - coil_idx: coil index
    '''

    # add these two slices together
    slice_data = np.add(ksp[0, slice_idx, coil_idx, :, :], ksp[1, slice_idx, coil_idx, :, :])

    print(f"slice data shape {slice_data.shape}")

    # Separate real and imaginary parts
    real_data = np.real(slice_data)
    imag_data = np.imag(slice_data)

    # Print the range of real and imaginary parts
    print(f"Real Part Range: {np.min(real_data)} - {np.max(real_data)}")
    print(f"Imaginary Part Range: {np.min(imag_data)} - {np.max(imag_data)}")
    
    # make two subplots that are histograms of the real and imaginary parts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Histogram for {title}")
    
    ax1.hist(real_data.flatten(), bins=100, color='blue', alpha=0.7, label="Real", range=(np.min(real_data)/50, np.max(real_data)/50))
    ax1.set_title("Real Part")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    
    ax2.hist(imag_data.flatten(), bins=100, color='red', alpha=0.7, label="Imaginary", range=(np.min(imag_data)/50, np.max(imag_data)/50))
    ax2.set_title("Imaginary Part")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    
    plt.show()


def normalize_kspace(kspace_data: np.ndarray) -> np.ndarray:
    '''
    Normalize k-space data by its maximum magnitude value and wrap phase values within [-π, π].
    args:
        kspace_data: k-space data in complex form
    returns:
        normalized_kspace: normalized k-space data in complex form

    '''
    # Compute magnitude and phase (angle)
    magnitude = np.abs(kspace_data)
    phase = np.angle(kspace_data)
    
    # Normalize magnitude by its maximum value
    normalized_magnitude = magnitude / np.max(magnitude)
    del magnitude
    
    # Ensure phase is wrapped within [-π, π]
    # Note: The numpy angle function already returns values in this range, 
    # but this step is added for completeness and in case phase values are modified elsewhere.
    wrapped_phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
    del phase
    
    # Convert normalized magnitude and wrapped phase back to complex form
    normalized_kspace = normalized_magnitude * np.exp(1j * wrapped_phase)
    del normalized_magnitude, wrapped_phase
    
    return normalized_kspace


def normalize_to_reference(
    ksp: np.ndarray,
    max_magni_ref: int,
    logger: logging.Logger
) -> np.ndarray:
    """
    Normalize k-space data to reference k-space data by its maximum magnitude value and wrap phase values within [-π, π].
    
    Parameters:
        ksp (np.ndarray): k-space data in complex form (2, slices, coils, freq, phase)
        max_magni_ref (int): maximum magnitude value of the reference k-space data
        logger (logging.Logger): The logger for logging messages.
    
    Returns:
        norm_to_ref_ksp: normalized k-space data in complex form
    """

    logger.info(f"\tNormalizing k-space data to reference {max_magni_ref} k-space data... (takes a while). Reference to mean NYU kspace data.")

    # # Compute maximum magnitude from the reference k-space
    # max_magni_ref = np.max(np.abs(max_magni_ref))
    
    # Scale the normalized magnitude by the reference's maximum magnitude
    # scaled_magnitude = np.abs(ksp) / np.max(np.abs(ksp)) * max_magni_ref        # simple version that can be understood
    
    # Ensure phase is wrapped within [-π, π]
    # wrapped_phase = np.mod(np.angle(ksp) + np.pi, 2 * np.pi) - np.pi                         # simple version that can be understood
    
    # Convert scaled magnitude and wrapped phase back to complex form
    # norm_to_ref_ksp = scaled_magnitude * np.exp(1j * wrapped_phase)  # simple version that can be understood
    # norm_to_ref_ksp = (np.abs(ksp) / np.max(np.abs(ksp)) * max_magni_ref) * np.exp(1j * (np.mod(np.angle(ksp) + np.pi, 2 * np.pi) - np.pi))

    # Made the whole function as 1 line for memory efficiency
    return (np.abs(ksp) / np.max(np.abs(ksp)) * max_magni_ref) * np.exp(1j * (np.mod(np.angle(ksp) + np.pi, 2 * np.pi) - np.pi))


def center_crop_im(im_3d: np.ndarray, crop_to_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop an image to a given size.
    
    Parameters:
    -----------
    im_3d : numpy.ndarray
        Input image of shape (slices, x, y).
    crop_to_size : list
        List containing the target size for x and y dimensions.
    
    Returns:
    --------
    numpy.ndarray
        Center cropped image of size {slices, x_cropped, y_cropped}. 
    """
    x_crop = im_3d.shape[2]/2 - crop_to_size[0]/2
    y_crop = im_3d.shape[1]/2 - crop_to_size[1]/2

    return im_3d[:, int(y_crop):int(crop_to_size[1] + y_crop), int(x_crop):int(crop_to_size[0] + x_crop)]  


def get_dicom_headers(fpath: str) -> object:
    '''
    Description:
        This function returns the dicom headers of a dicom file.
    Args:
        fpath (str): The path to the dicom file.
    Returns:
        ds (object): The dicom headers.
    '''
    hdrs_object = pydicom.dcmread(fpath)
    return hdrs_object


def create_h5_if_not_exists(fpath_hf: str) -> None:
    '''
    Description:
        This function creates an h5 file if it does not exist.
    Args:
        fpath_hf (str): The path to the h5 file.
    '''
    if not os.path.exists(fpath_hf):
        with h5py.File(fpath_hf, 'w') as hf:
            print(f"\tcreated h5 file at {fpath_hf}")
    else:
        print(f"\tH5 file already exists: {fpath_hf}")


# def get_anon_id_from_seq_id(pat_seq_id: str, pat_map_df: pd.DataFrame):
#     '''
#     Args:
#         pat_seq_id (str): The patient id of the sequence.
#     Returns:
#         pat_anon_id (str): The anonymized patient id.
#     '''
#     return pat_map_df[pat_map_df['pat_seq_id'] == pat_seq_id]['pat_anon_id'].iloc[0]


def get_t2_tra_mrd_fname(target_dir: Path, logger: logging.Logger) -> Path:
    '''
    This function returns the path to the t2 tra mrd file.
    
    Parameters:
        target_dir (Path): The path to the patient dir.
        logger (logging.Logger): The logger for logging messages.
    
    Returns:
        fpath (Path): The path to the t2 tra mrd file.
    '''
    paths = []
    for fpath in target_dir.glob('*'):
        if fpath.name.endswith('2.mrd'):
            paths.append(fpath)
            logger.info(f"\tfound t2 tra mrd path: {fpath}")
    
    # if not exactly one is found we have a problem, then log and raise an error
    if len(paths) != 1:
        logger.error(f"found {len(paths)} t2 tra mrd paths: {paths}")
        raise ValueError(f"found {len(paths)} t2 tra mrd paths: {paths}")
    path = paths[0]
    return path


#################### k-space conversion essentials ####################


def find_t2_tra_kspace_files(patient_kspace_directory: Path, logger: logging.Logger, verbose: bool = False) -> List[Path]:
    """Finds and returns T2-weighted transverse k-space .dat files in a specified directory.

    Args:
        patient_kspace_directory (Path): The directory containing k-space .dat files.
        logger (logging.Logger): The logger for logging messages.
        verbose (bool, optional): Whether to print verbose log messages. Defaults to False.

    Returns:
        List[Path]: A list of unique k-space .dat files as Path objects.
    """
    search_patterns = [
        'meas_MID*_T2_TSE_tra*.dat',
        'meas_MID*_t2_tse_tra*.dat'
    ]

    unique_file_paths = set()
    for pattern in search_patterns:
        search_path = str(patient_kspace_directory / pattern)
        unique_file_paths.update(glob.glob(search_path))

    if not unique_file_paths:
        logger.warning(f"Could not find any .dat files in {patient_kspace_directory}")

    # Convert each unique file path string to a Path object
    unique_file_paths = {Path(file_path) for file_path in unique_file_paths}

    if verbose:
        for file_path in unique_file_paths:
            logger.info(f"Found {file_path}")

    return list(unique_file_paths)


def anonymize_mrd_files(
    pat_out_dir: Path,
    pat_kspace_dir: Path,
    logger: logging.Logger,
    mrd_xml: Path,
    mrd_xsl: Path
) -> bool:
    """
    Function for .dat to .mrd anonymization process using siemens_to_ismrmrd.
    """
    try:
        dat_files = find_t2_tra_kspace_files(pat_kspace_dir, logger, verbose=False)
        output_dir = pat_out_dir / "mrds"
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_file in dat_files:
            output_file = output_dir / f"{input_file.stem}-out.mrd"

            if output_file.exists():
                continue  # Skip if the output file already exists
            
            cmd = [
                # "wsl",
                "siemens_to_ismrmrd",
                "-f", input_file,
                "-m", mrd_xml,
                "-x", mrd_xsl,
                "-o", output_file,
                "-Z"
            ]
            subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        logger.info(f"Successfully anonymized {len(dat_files)} files.")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred during the conversion process: {e}")
        return False


def connect_to_database(database_path: str):
    """
    Connect to an SQLite database.
    
    Parameters:
    - database_path (str): Path to the SQLite database file.
    
    Returns:
    - conn (sqlite3.Connection): SQLite database connection object.
    """
    conn = sqlite3.connect(database_path)
    return conn


def setup_logger(log_dir: Path, use_time: bool = True) -> logging.Logger:
    """
    Configure logging to both console and file.
    This function sets up logging based on the specified logging directory.
    It creates a log file named with the current timestamp and directs log 
    messages to both the console and the log file.
    Parameters:
    - log_dir (Path): Directory where the log file will be stored.
    Returns:
    - logging.Logger: Configured logger instance.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if use_time: 
        log_file = log_dir / f"log_{current_time}.log"
    else:
        log_file = log_dir / "log.log"

    l = logging.getLogger()
    l.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    l.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    l.addHandler(console_handler)

    return l


def create_h5_if_not_exists(fpath_hf: str, logger: logging.Logger) -> None:
    '''
    Description:
        This function creates an h5 file if it does not exist.
    Args:
        fpath_hf (str): The path to the h5 file.
    '''
    if not os.path.exists(fpath_hf):
        with h5py.File(fpath_hf, 'w') as hf:
            logger.info(f"\tcreated h5 file at {fpath_hf}")
    else:
        logger.info(f"\tH5 file already exists: {fpath_hf}")


def build_kspace_array_from_mrd_umcg(fpath_mrd: str, logger: logging.Logger = None) -> np.ndarray:
    """
    This function builds a k-space array from a .mrd file.
    
    Parameters:
    - fpath_mrd (str): Path to the .mrd file.
    - verbose (bool): Whether to print verbose log messages.
    - logger (logging.Logger): The logger for logging messages.
    
    Returns:
    - kspace (numpy.ndarray): The k-space data.
    """
    
    logger.info(f"\tBuilding kspace array from .mrd file")

    # Read the header and get encoding information
    dset   = ismrmrd.Dataset(fpath_mrd, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc    = header.encoding[0]

    # Determine some parameters of the acquisition
    ncoils     = header.acquisitionSystemInformation.receiverChannels
    nslices    = enc.encodingLimits.slice.maximum + 1 if enc.encodingLimits.slice is not None else 1
    eNy        = enc.encodedSpace.matrixSize.y
    rNx        = enc.reconSpace.matrixSize.x
    # eTL        = 25 if DEBUG else echo_train_length(dset)
    # eTC        = 11 if DEBUG else echo_train_count(dset, echo_train_len=eTL)
    firstacq   = get_first_acquisition(dset)
    navgs      = 3 #if DEBUG else get_num_averages(firstacq=firstacq, dset=dset)

    # Loop through the rest of the acquisitions and fill the data array with the kspace data
    kspace = np.zeros((navgs, nslices, ncoils, rNx, eNy + 1), dtype=np.complex64)
    logger.info(f"\tFilling kspace array from mrd object to shape {kspace.shape}...\n\tNum Acquisitions: {dset.number_of_acquisitions()} \n\t\tLoading... ")

    for acqnum in range(firstacq, dset.number_of_acquisitions()):
        if acqnum % 1000 == 0:
            print(f"{acqnum/dset.number_of_acquisitions() * 100:.0f}%", end=" ", flush=True)
        
        acq    = dset.read_acquisition(acqnum)
        slice1 = acq.idx.slice
        y      = acq.idx.kspace_encode_step_1
        avg    = acq._head.idx.average

        # Each acquisition is a 2D array of shape (coil, rNx) complex
        kspace[avg, slice1, :, :, y] = acq.data

    print()
    return kspace


def reorder_k_space_even_odd(ksp: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """
    Rearranges the k-space data by interleaving even and odd indexed slices.

    Parameters:
    - ksp (numpy.ndarray): The input k-space data to be rearranged.

    Returns:
    - interleaved_ksp (numpy.ndarray): The k-space data with interleaved slices.
    """
    # assert that the kspace is 5D
    assert len(ksp.shape) == 5, f"kspace is not 5D. Shape: {ksp.shape}, shape should be (navgs, nslices, ncoils, n_freq, n_phase)"

    # Initialize a new complex array to store the interleaved k-space data
    interleaved_ksp = np.zeros_like(ksp, dtype=np.complex64)

    # Calculate the middle index for slicing the array into two halves
    num_slices = ksp.shape[1]
    middle_idx = (num_slices + 1) // 2  # Handles both odd and even cases

    logger.info(f"\tReordering k-space by interleaving even and odd indexed slices, num_slices: {num_slices}, middle_idx: {middle_idx} is_even: {num_slices % 2 == 0}")

    # Interleave even and odd indexed slices, for some reason it depends on being even or odd.
    if num_slices % 2 == 0: # Even number of slices
        ksp = np.flip(ksp, axis=1)
        interleaved_ksp[:, ::2] = ksp[:, middle_idx:]  # Place the second half at even indices
        interleaved_ksp[:, 1::2] = ksp[:, :middle_idx]  # Place the first half at odd indices
    else: # Odd number of slices
        interleaved_ksp[:, ::2]  = ksp[:, :middle_idx]
        interleaved_ksp[:, 1::2] = ksp[:, middle_idx:]
        interleaved_ksp = np.flip(interleaved_ksp, axis=1)

    return interleaved_ksp


def remove_zero_padding(kspace: np.ndarray, logger: logging.Logger = None) -> np.ndarray:
    """
    Remove the zero padding in the phase encoding direction from the given k-space.
    
    Parameters:
        kspace (np.ndarray): The k-space data. Should have 5 dimensions: (n_avg, n_slices, n_coils, n_freq, n_phase)
    
    Returns:
        np.ndarray: The k-space data without zero padding in the phase encoding direction.
    """
    assert kspace.ndim == 5, "image should have 5 dimensions: (n_avg, n_slices, n_coils, n_freq, n_phase)"
    logger.info(f"\tkspace shape before zero-padding removal: {kspace.shape}")

    n_avg, n_slices, n_coils, n_freq, n_phase = kspace.shape
    zero_padding, idxs = calculate_zero_padding_PE(kspace)
    logger.info(f"\tFound zero padding of {zero_padding} in phase encoding direction.")

    # Remove the zero padding in the phase encoding direction
    return kspace[:, :, :, :, 0:n_phase - zero_padding]


def get_first_acquisition(dset) -> int:
    '''
    Arguments:
        - dset: ismrmrd.Dataset object
    Returns:
        - firstacq: index of the first acquisition
    '''
    firstacq = 0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            firstacq = acqnum
            print("\tImaging acquisition starts at acq: ", acqnum)
            break
    return firstacq


def crop_kspace_in_phase_direction(
        kspace: np.ndarray,
        max_phase_crop: int,
        fpath_mrd: str,
        logger: logging.Logger = None
) -> np.ndarray:
    """
    Crop the k-space in the phase direction to achieve the desired target shape.
    
    Arguments:
    - ksp: 5D numpy array of shape (navgs, nslices, ncoils, read, phase) complex.
    - max_phase: Maximum phase integer value for cropping. in the phase direction
    - fpath_mrd: Path to the .mrd file. This is needed to change the headers.
    - verbose: Print the cropping details.

    Returns:
    - Cropped k-space numpy array.
    """
    
    # Check input shape validity
    if len(kspace.shape) != 5:
        raise ValueError("ksp must be 5D (navgs, nslices, ncoils, read, phase) complex.")
    if kspace.shape[-1] < max_phase_crop:
        raise ValueError("The k-space phase dimension should be smaller than the desired phase crop.")
    
    # the headers of the kspace must be changed if you do a phase cropping. This is read from the MRD file.
    new_hdrs = change_headers_based_on_phase_cropping(fpath_mrd, max_phase_int=max_phase_crop)
    
    if kspace.shape[-1] == max_phase_crop:
        logger.info("\tKspace and desired phase shape are equal so return kspace as is.")
        return kspace, new_hdrs

    # Calculate the cropping size in the phase direction
    phase_crop_size = kspace.shape[-1] - max_phase_crop
    left_crop       = phase_crop_size // 2
    right_crop      = phase_crop_size - left_crop

    if logger:
        logger.info(f"Original k-space shape: {kspace.shape}")
        cur_shape = list(kspace.shape)
        cur_shape[-1] -= phase_crop_size
        logger.info(f"Cropped kspace will be: {tuple(cur_shape)}")

    # Return the cropped k-space and the new headers
    return kspace[..., left_crop:-right_crop], new_hdrs


def change_headers_based_on_phase_cropping(fpath_mrd: str, max_phase_int: int) -> bytes:
    """
    Adjust the headers of an .mrd file to match the NYU format based on phase cropping.

    Parameters:
    - fpath_mrd: Path to the .mrd file.
    - max_phase_int: Maximum phase integer value for cropping.

    Returns:
    - header_bytes: Transformed headers in byte format.
    """

    # Namespace string for ISMRMRD XML
    ns = "{http://www.ismrm.org/ISMRMRD}"

    # Retrieve the headers from the .mrd file
    umcg_headers_mrd = get_headers_from_ismrmrd(fpath_mrd, verbose=False)

    # Convert the headers to a dictionary
    umcg_headers_dict = convert_ismrmrd_headers_to_dict(umcg_headers_mrd)

    # Update headers with the correct matrix size and encoding limits for NYU data format
    umcg_headers_dict[f"{ns}ismrmrdHeader"][f"{ns}encoding"][f"{ns}encodedSpace"][f"{ns}matrixSize"][f"{ns}y"] = str(max_phase_int)
    umcg_headers_dict[f"{ns}ismrmrdHeader"][f"{ns}encoding"][f"{ns}encodingLimits"][f"{ns}kspace_encoding_step_1"][f"{ns}maximum"] = str(max_phase_int)
    umcg_headers_dict[f"{ns}ismrmrdHeader"][f"{ns}encoding"][f"{ns}encodingLimits"][f"{ns}kspace_encoding_step_1"][f"{ns}center"] = str(max_phase_int//2)

    header_bytes = encode_umcg_header_to_bytes(umcg_to_nyu_dict=umcg_headers_dict)
    return header_bytes
