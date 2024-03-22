import os
import h5py
import logging
import sqlite3
import pydicom
import numpy as np
from typing import Dict
from pathlib import Path

from assets.reconstruction import build_kspace_array_from_mrd_umcg
from assets.operations_kspace import reorder_k_space_even_odd, remove_zero_padding, normalize_to_reference
from assets.convert_to_mrd import get_headers_from_ismrmrd, convert_ismrmrd_headers_to_dict, encode_umcg_header_to_bytes
from assets.reconstruction import safe_rss_to_nifti_file
from assets.util import get_t2_tra_mrd_fname


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


def convert_mrd_to_h5(
    pat_dir: Path,
    study_date: str,
    logger: logging.Logger,
    conn: sqlite3.Connection, 
    tablename_ksp: str = None,
    tablename_dcm: str = None,
    **kwargs,
) -> None:
    """
    Converts MRD files to H5 format if not already done.
    Skips conversion if 'has_h5' is already set for the patient.
    
    Parameters:
    pat_dir (Path): Path object pointing to the patient directory.
    study_date (str): The study date of the patient.
    logger (logging.Logger): Logger object for logging messages.
    conn (sqlite3.Connection): SQLite database connection object.
    tablename (str): Name of the database table to check and update. Defaults to 'kspace_dset_info'.
    
    raises:
    sqlite3.Error: If an error occurs during database operations.
    """
    assert tablename_ksp is not None, "Table name for kspace patient info must be provided."
    assert tablename_dcm is not None, "Table name for DICOM patient info must be provided."
    
    seq_id, anon_id = pat_dir.name.split('_')

    try:
        with conn:
            cur = conn.cursor()
            # Check if conversion is already done
            cur.execute(f"SELECT has_h5 FROM {tablename_ksp} WHERE seq_id = ?", (seq_id,))
            result = cur.fetchone()
            if result and result[0]:
                logger.info(f"\tConversion already done for patient {seq_id}. Skipping.")
                return
            
            logger.info(f"\tConverting .mrd files to .h5 files for patient in {pat_dir}")
            perform_conversion(
                pat_dir       = pat_dir,
                seq_id        = seq_id,
                anon_id       = anon_id,
                study_date    = study_date,
                tablename_ksp = tablename_ksp,
                tablename_dcm = tablename_dcm,
                conn          = conn,
                logger        = logger,
                **kwargs
            )
    except sqlite3.Error as e:
        logger.error(f"Database error during conversion check: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")


def perform_conversion(
    pat_dir,
    seq_id: str,
    anon_id: str,
    study_date: str,
    tablename_ksp: str,
    tablename_dcm: str,
    conn: sqlite3.Connection,
    logger: logging.Logger,
    do_rm_zero_pad: bool = True,
    do_norm_to_ref: bool = True,
    max_phase_crop: int = None,
    max_mag_ref: float = 0.010586672,
    **kwargs
) -> None:
    """
    Performs the actual conversion of MRD files to H5 format.
    
    Parameters:
    pat_dir (Path): Path object pointing to the patient directory.
    seq_id (str): The sequence ID of the patient.
    anon_id (str): The anonymized ID of the patient. 
    study_date (str): The study date of the patient.
    tablename_ksp (str): Name of the database table for kspace patient info.
    tablename_dcm (str): Name of the database table for DICOM patient info.
    conn (sqlite3.Connection): SQLite database connection object.
    logger (logging.Logger): Logger object for logging messages.
    do_rm_zero_pad (bool): If True, the zero padding is removed.
    do_norm_to_ref (bool): If True, the magnitude is normalized to the reference magnitude.
    max_phase_crop (int): The maximum phase integer value for cropping.
    max_mag_ref (float): The reference magnitude value. Defaults to 0.010586672.
    
    Raises:
    Exception: If an error occurs during conversion.
    """
    mrd_fpath = get_t2_tra_mrd_fname((pat_dir / 'mrds'), logger)
    fpath_hf = Path(pat_dir / 'h5s', f"{mrd_fpath.stem}.h5")
    create_h5_if_not_exists(fpath_hf, logger)
    
    # Add the K-space to the H5 file.
    with h5py.File(fpath_hf, 'r+') as hf:
        if not has_key_in_h5(fpath_hf, 'kspace', logger):
            kspace, headers = convert_mrd_to_array(
                fpath_mrd        = mrd_fpath,
                pat_rec_dir      = pat_dir / 'recons',
                max_mag_ref      = max_mag_ref,  # Entire NYU test and validation dataset # one patient: 0.006096669
                do_rm_zero_pad   = do_rm_zero_pad, 
                do_norm_to_ref   = do_norm_to_ref,
                max_phase_crop   = max_phase_crop,         # None means that the kspace is not cropped in the phase dir.
                logger           = logger,
            )
            
            hf.create_dataset('ismrmrd_header', data=headers)
            hf.create_dataset('kspace', data=kspace)
            logger.info(f"\tCreated 'kspace' dataset and 'ismrmrd_header' in {fpath_hf}")

        if not has_correct_shape(fpath_hf, logger):
            logger.error(f"kspace shape is not correct. Shape: {hf['kspace'].shape}")
            raise Exception(f"kspace shape is not correct. Shape: {hf['kspace'].shape}")

    # Read the first file in the pat_dcm_dir to get the dcm headers. 
    dcm_hdrs = extract_t2_tra_metadata(pat_dir, study_date, logger, conn, tablename_dcm)
    
    with h5py.File(fpath_hf, 'r+') as hf:  # Add the attributes to the H5 file.
        if len(dict(hf.attrs)) == 0:
            hf.attrs['acquisition']    = 'AXT2'
            hf.attrs['max']            = 0.0004        # Chosen as the mean/median value from the NYU dataset.
            hf.attrs['norm']           = 0.12          # Chosen as the mean/almost_median value from the NYU dataset.
            hf.attrs['patient_id']     = anon_id
            hf.attrs['patient_id_seq'] = seq_id
            hf.attrs['do_rm_zero_pad'] = do_rm_zero_pad
            hf.attrs['do_norm_to_ref'] = do_norm_to_ref
            hf.attrs['max_phase_crop'] = 'None' if max_phase_crop is None else str(max_phase_crop)
            for key in dcm_hdrs.keys():
                hf.attrs[key + "_dcm_hdr"] = dcm_hdrs[key]
            logger.info(f"\tAdded attributes to h5")
    
    with h5py.File(fpath_hf, 'r') as hf: # Log the attributes of the H5 file.
        for key in dict(hf.attrs).keys():
            logger.info(f"\t\t{key}: {hf.attrs[key]}")

    try:  # Update the database at the end
        with conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE {tablename_ksp} SET has_h5 = 1 WHERE seq_id = ?", (seq_id,))
            logger.info(f"\tUpdated {tablename_ksp} for patient {seq_id} to indicate successful H5 conversion.")
    except sqlite3.Error as e:
        logger.error(f"Failed to update database for patient {seq_id} with error: {e}")
        

def convert_mrd_to_array(
    fpath_mrd: str,
    pat_rec_dir: str,
    max_mag_ref: float,
    do_rm_zero_pad: bool,
    do_norm_to_ref: bool,
    max_phase_crop: int = None,
    logger: logging.Logger = None
) -> None:
    '''
        This function converts a .mrd file to a numpy array.
        The kspace is cropped in the phase direction to the shape of the NYU dataset.
        The kspace is normalized to the reference magnitude of the NYU dataset.
    Parameters:
        fpath (str): The path to the .mrd file.
        phase_crop_shape (tuple): The shape to crop the kspace to.
        max_mag_ref (float): The reference magnitude.
        do_rm_zero_pad (bool): If True, the zero padding is removed.
        do_norm_to_ref (bool): If True, the magnitude is normalized to the reference magnitude.
    Returns:
        kspace (np.ndarray): The kspace array.
        trans_hdrs (dict): The transformed headers.
    '''
    
    # Construct the kspace array from the sequentail MRD object.
    kspace = build_kspace_array_from_mrd_umcg(fpath_mrd, logger)

    # Reorder the slices of the kspace based on even and odd number of slices
    kspace = reorder_k_space_even_odd(kspace, logger)

    # Remove the zero padding from the kspace.
    if do_rm_zero_pad:
        kspace = remove_zero_padding(kspace, logger)

    if max_phase_crop == None:
        max_phase_crop = kspace.shape[-1]

    # Crop the kspace in the phase dir and obtain the transformed headers. Simply extracts the headers as is, if the crop shape is equal to the kspace shape.
    kspace, trans_hdrs = crop_kspace_in_phase_direction(kspace, max_phase_crop, fpath_mrd, logger)

    if do_norm_to_ref:
        try:
            kspace = normalize_to_reference(kspace, max_mag_ref, logger)
        except Exception as e:
            logger.error(f"Error in normalizing kspace: {e}")
            raise Exception(f"Error in normalizing kspace: {e}")

    safe_rss_to_nifti_file(kspace=kspace, fname_part="pre_processed_ksp", do_round=True, dir=pat_rec_dir, logger=logger)

    return kspace, trans_hdrs


# def create_h5_if_not_exists(fpath_hf: str) -> None:
#     '''
#     Description:
#         This function creates an h5 file if it does not exist.
#     Args:
#         fpath_hf (str): The path to the h5 file.
#     '''
#     if not os.path.exists(fpath_hf):
#         with h5py.File(fpath_hf, 'w') as hf:
#             print(f"\tcreated h5 file at {fpath_hf}")
#     else:
#         print(f"\tH5 file already exists: {fpath_hf}")

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
    

def extract_t2_tra_metadata(
    pat_dir: Path,
    study_date: str,
    logger: logging.Logger = None,
    conn: sqlite3.Connection = None,
    tablename: str = None
) -> Dict:
    """
    Extracts metadata from DICOM headers of T2 TRA TSE MRI sequence.

    Parameters:
    pat_dir (Path): Directory of the patient's data.
    study_date (str): Date of the study.
    logger (logging.Logger): Logger for error messages.
    tablename (str): Name of the database table.

    Returns:
    Dict: Dictionary containing extracted metadata.
    """
    assert tablename is not None, "Table name must be provided."
    assert conn is not None, "Database connection must be provided."
    
    seq_id, anon_id = pat_dir.name.split('_')
    dicoms_dir = Path(pat_dir / 'dicoms' / study_date)
    cur = conn.cursor()
    
    sd = study_date.replace('-', '')
    
    try:
        query = f"""
            SELECT NiftiPath FROM {tablename}
            WHERE patientID LIKE ?
            AND ProtocolName LIKE "%T2%"
            AND ProtocolName LIKE "%tra%"
            AND ProtocolName NOT LIKE "%blade%"
            AND ProtocolName NOT LIKE "%bekken%"
            AND ProtocolName NOT LIKE "%hele buik%"
            AND NiftiPath NOT NULL
            AND SeriesDate = {sd}
            ORDER BY AcquisitionTime DESC
            LIMIT 1;
        """
        cur.execute(query, (f"%{anon_id}%",))
        result = cur.fetchone()
        if not result:
            msg = f"No NIfTI path found for patient {anon_id}."
            logger.error(msg) if logger else print(msg)
            raise ValueError(msg)

        # Extract file stem from NIfTI path
        nifti_stem = Path(result[0]).stem
        nifti_stem = Path(nifti_stem).stem

        # Locate the DICOM file in dicoms_dir
        dicom_t2_dir = dicoms_dir / nifti_stem
        first_dcm_file = next(dicom_t2_dir.glob('*'))
        ds = pydicom.dcmread(first_dcm_file)

        info_dict = {
            'percent_phase_fov': ds.PercentPhaseFieldOfView,
            'n_phase_enc_steps': ds.NumberOfPhaseEncodingSteps,
            'percent_sampling': ds.PercentSampling,
            'acq_mat': ds.AcquisitionMatrix,
            'n_averages': ds.NumberOfAverages,
            'rows': ds.Rows,
            'cols': ds.Columns,
            'pat_pos': ds.PatientPosition,
            'pixel_spacing': ds.PixelSpacing,
        }
    
        logger.info(f"\tExtracted DICOM metadata: {info_dict}")

        return info_dict

    except StopIteration as e:
        msg = f"No DICOM file found in {dicom_t2_dir}."
        logger.error(msg) if logger else print(msg)
        raise ValueError(msg)