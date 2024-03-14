import os
import h5py
import logging
import sqlite3
from pathlib import Path

from assets.operations_kspace import build_kspace_array_from_mrd_umcg, reorder_k_space_even_odd, remove_zero_padding, crop_kspace_in_phase_direction, normalize_to_reference
from assets.reconstruction import safe_rss_to_nifti_file
from assets.util import get_t2_tra_mrd_fname


def convert_mrd_to_h5(
    pat_dir: Path,
    study_date: str,
    logger: logging.Logger,
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection, 
    tablename: str = 'kspace_dset_info'
) -> None:
    """
    Converts MRD files to H5 format if not already done.
    Skips conversion if 'has_h5' is already set for the patient.
    """
    seq_id, anon_id = pat_dir.name.split('_')

    # Check if conversion is already done
    cur.execute(f"SELECT has_h5 FROM {tablename} WHERE seq_id = ?", (seq_id,))
    if cur.fetchone()[0]:
        logger.info(f"\tConversion already done for patient {seq_id}. Skipping.")
        return

    logger.info(f"\tConverting .mrd files to .h5 files for patient in {pat_dir}")
    perform_conversion(pat_dir, study_date, logger, cur, conn, tablename, seq_id, anon_id)


def perform_conversion(
    pat_dir,
    study_date: str,
    logger: logging.Logger,
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection,
    tablename: str,
    seq_id: str,
    anon_id: str
) -> None:
    """
    Performs the actual conversion of MRD files to H5 format.
    
    Parameters:
    pat_dir (Path): Directory of the patient's data.
    study_date (str): Date of the study.
    logger (logging.Logger): Logger for error messages.
    cur (sqlite3.Cursor): Cursor object for executing SQL queries.
    conn (sqlite3.Connection): Connection object for connecting to the database.
    tablename (str): Name of the database table. the kspace patient info table
    seq_id (str): Sequential ID of the patient.
    anon_id (str): Anonymized ID of the patient.
    
    Raises:
    Exception: If an error occurs during conversion.
    """
    do_rm_zero_pad = True
    do_norm_to_ref = True
    max_phase_crop = None
    
    mrd_fpath = get_t2_tra_mrd_fname((pat_dir / 'mrds'), logger)
    fpath_hf = Path(pat_dir / 'h5s', f"{mrd_fpath.stem}.h5")
    create_h5_if_not_exists(fpath_hf, logger)
    
    # Add the K-space to the H5 file.
    with h5py.File(fpath_hf, 'r+') as hf:
        if not has_key_in_h5(fpath_hf, 'kspace', logger):
            kspace, headers = convert_mrd_to_array(
                fpath_mrd        = mrd_fpath,
                pat_rec_dir      = pat_dir / 'recons',
                max_mag_ref      = 0.010586672,  # Entire NYU test and validation dataset # one patient: 0.006096669
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
    dcm_hdrs = extract_t2_tra_metadata(pat_dir, study_date, logger, cur)
    
    # Add the attributes to the H5 file.
    with h5py.File(fpath_hf, 'r+') as hf:
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
    
    # Log the attributes of the H5 file.
    with h5py.File(fpath_hf, 'r') as hf:
        for key in dict(hf.attrs).keys():
            logger.info(f"\t\t{key}: {hf.attrs[key]}")

    # Update the database at the end
    if conn and cur:
        cur.execute(f"UPDATE {tablename} SET has_h5 = 1 WHERE seq_id = ?", (seq_id,))
        conn.commit()
        logger.info(f"\tUpdated {tablename} for patient {seq_id} to indicate successful H5 conversion.")
        

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