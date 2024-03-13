import argparse
import logging
from pathlib import Path
import sqlite3
from typing import Tuple, List, Dict
import subprocess
import glob
import pydicom
import h5py
import pandas as pd

from helper_functions import *


def parse_args(verbose: bool = False):
    """
    Parse command line arguments.
    Parameters:
    - verbose (bool): Flag for printing command line arguments to console.
    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Make kspace dataset ready for training.")

    # Input directory of the raw data
    parser.add_argument("-fmri1",
                        "--fastmri1_ksp_fpath",
                        type=str,
                        default="/mnt/d/01_data/01_prostate_raw_kspace_umcg",         # this is the FASTMRI1 drive, EXTERNAL SSD (D:)
                        help="Path to kspace directory.")

    # Output directory of the processed data
    parser.add_argument("-fmri2",
                        "--fastmri2_out_fpath",
                        type=str,
                        default="/mnt/e/kspace/02_umcg_pst_ksps_processed",     # this is the FASTMRI2 drive, external SSD  (E:)
                        help="Path to target data directory.")
    
    # Root dir of the UMCG library
    parser.add_argument("-umcglib",
                        "--umcglib_dir",
                        type=str,
                        default="/mnt/c/users/lohuizenqy/local_docs/repos/umcglib",     # this is the FASTMRI2 drive, external SSD  (E:)
                        help="Path to umcglib directory.")
    
    # Anonymized dicoms directory (processed by the Datamanager C# program)
    parser.add_argument("-dd",
                        "--anon_dcms_dir",
                        type=str,
                        default="/mnt/e/kspace/02_umcg_pst_ksps_processed/data/sectra_dynacad_exp_dcms_and_rois_pat1_136_anon_dcms",
                        help="Path to anonymized dicoms directory. Processed by the Datamanager C# program.")
    
    # Nifti directory of the dicoms and rois by the Datamanager C# program
    parser.add_argument("-nd",
                        "--nifti_dir",
                        type=str,
                        default="/mnt/e/kspace/02_umcg_pst_ksps_processed/data/sectra_dynacad_exp_dcms_and_rois_pat1_136_niftis",
                        help="Path to nifti directory. Processed by the Datamanager C# program.")


    parser.add_argument("-th",
                        "--transfer",
                        action="store_true",
                        default = True,
                        help="Debug mode.")


    parser.add_argument("--debug",
                        action="store_true",
                        help="Debug mode.")
    
    a = parser.parse_args()

    if verbose:
        print("Command line arguments:")
        for arg in vars(a):
            print(f"- {arg}:\t\t{getattr(a, arg)}")

    return a


def filter_and_sort_patient_dirs(target_dir: Path) -> list:
    logging.info("Filtering and sorting patient directories.")
    dirs = [f.name for f in target_dir.iterdir() if f.is_dir()]
    dirs = [f for f in dirs if f.split('_')[1] == 'patient']
    dirs.sort(key=lambda x: int(x.split('_')[0]))
    return dirs


def create_directory_structure(base_path: Path):
    sub_dirs = ['lesion_bbs', 'recons', 'dicoms', 'mrds', 'h5s', 'rois', 'niftis']

    for sub_dir in sub_dirs:
        (base_path / sub_dir).mkdir(parents=True, exist_ok=True)


def create_and_log_patient_dirs(kspace_dir: Path, out_dir: Path, debug: bool, logger: logging.Logger) -> None:
    """
    Process patient directories to create the required directory structure and
        log relevant information.
    Parameters:
    - kspace_dir (Path): The directory containing the k-space data.
    - out_dir (Path): The directory where the processed data will be stored.
    - debug (bool): Flag for debugging. If True, the loop breaks after 2 iterations.
    - logger (logging.Logger): Logger object for logging information.
    """
    pat_dirs = filter_and_sort_patient_dirs(kspace_dir)

    for pat_idx, pat_dir in enumerate(pat_dirs, start=1):
        pat_dir_path = kspace_dir / pat_dir
        dicom_dir_path = pat_dir_path / 'dicoms'

        assert len([f.name for f in dicom_dir_path.iterdir() if f.is_dir()]) == 1, \
            "There should be only one dir in the dicom dir"

        pat_dcm_dir = [f for f in dicom_dir_path.iterdir() if f.is_dir()][0]
        pat_num = pat_dcm_dir.name.split('_')[-1]
        anon_pat_id = encipher(pat_num, key=KEY)  # Assuming encipher is defined elsewhere
        seq_id = str(pat_idx).zfill(4)

        new_pat_dir_path = out_dir / f"{seq_id}_{anon_pat_id}"

        if not new_pat_dir_path.exists():
            create_directory_structure(new_pat_dir_path)  # Assuming this function is defined elsewhere
            logger.info("Created new directory: %s", new_pat_dir_path)
        else:
            # logger.info("Directory already exists: %s", new_pat_dir_path)
            pass

        if debug and pat_idx > 2:
            break


def get_patient_info(target_dir: Path, logger: logging.Logger, verbose: bool = False) -> Tuple[str, str, Path]:
    """
    Retrieve and sort patient directories in a given target directory based on the patient number.

    Parameters:
    - target_dir (Path): The target directory containing patient directories.
    - verbose (bool): If True, print the sorted patient directories.

    Returns:
    - List[Tuple[str, str, Path]]: A list of tuples, each containing:
        - Patient number (str)
        - anon_id (str)
        - Full directory path (Path)
    """
    # Filter directories and sort them based on the patient number.
    pat_dirs = [f for f in target_dir.iterdir() if f.is_dir() and len(f.name.split('_')[0]) == 4]
    pat_dirs.sort(key=lambda x: int(x.name.split('_')[0]))

    # Create tuples from sorted directories.
    pat_info = [(f.name.split('_')[0], f.name.split('_')[1], f) for f in pat_dirs]

    # Verbose logging
    if verbose:
        for pat_number, anon_id, full_path in pat_info:
            logger.info(f"Patient Number: {pat_number}, anon_id: {anon_id}, Full Path: {full_path}")

    return pat_info


def update_patient_info_in_db(
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection,
    table: str,
    seq_id: str,
    anon_id: str,
    pat_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Update patient information in SQLite database.

    Parameters:
    cur: SQLite cursor object.
    conn: SQLite connection object.
    table (str): Name of the table in the database.
    seq_id: Sequential ID of the patient.
    anon_id: Anonymized ID of the patient.
    pat_dir: Directory path of patient's data.
    logger: Logger object for logging messages.
    """
    # Use parameterized queries to avoid SQL injection
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE seq_id = ?", (seq_id,))
    if cur.fetchone()[0] == 0:
        # Insert a new row if this patient is not in the database
        cur.execute(f"""
            INSERT INTO {table} (seq_id, anon_id, data_dir)
            VALUES (?, ?, ?)
        """, (seq_id, anon_id, str(pat_dir)))
        logger.info(f"Inserted new patient info in database for patient {seq_id}.")
    else:
        # Update existing row for this patient
        cur.execute(f"""
            UPDATE {table}
            SET anon_id = ?, data_dir = ?
            WHERE seq_id = ?
        """, (anon_id, str(pat_dir), seq_id))
        logger.info(f"\tUpdated patient info in database for patient {seq_id}.")
    conn.commit()
    

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


def process_mrd_if_needed(
    cur: sqlite3.Cursor,
    pat_out_dir: Path,
    raw_ksp_dir: Path,
    logger: logging.Logger,
    mrd_xml: Path,
    mrd_xsl: Path,
    tablename: str  # New parameter for the table name
) -> None:
    """
    Perform .dat to .mrd anonymization if not done already.

    Parameters:
    cur: SQLite cursor object.
    pat_out_dir: Path object for patient output directory.
    raw_ksp_dir: Path object for raw k-space directory.
    logger: Logger object for logging messages.
    mrd_xml: Path object for MRD XML file.
    mrd_xsl: Path object for MRD XSL file.
    tablename (str): Name of the table in the database.
    """
    cur.execute(f"SELECT has_mrds FROM {tablename} WHERE data_dir = ?", (str(pat_out_dir),))
    has_mrd = cur.fetchone()[0]

    if has_mrd is None or has_mrd == 0:
        success = anonymize_mrd_files(pat_out_dir, raw_ksp_dir, logger, mrd_xml, mrd_xsl)  # Implement this function

        if success:
            cur.execute(f"""
                UPDATE {tablename}
                SET has_mrds = 1
                WHERE data_dir = ?
            """, (str(pat_out_dir),))

            
            
def write_pat_info_to_file(patients, logger, out_dir):
    """	
    Write patient info to a csv file.
    """
    df = pd.DataFrame(columns=['seq_id', 'id', 'anon_id', 'data_dir'])
    new_rows = []

    for seq_id, anon_id, pat_dir in patients:
        id = decipher(anon_id, key=KEY)
        
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


def copy_nifti_files_if_needed(
    anon_id: str,
    nifti_src_dir: Path,
    pat_dir: Path,
    logger: logging.Logger,
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection,
    tablename: str = 'kspace_dset_info'
) -> None:
    """
    Copies NIfTI files from the source directory to the patient's NIfTI directory.

    Parameters:
    anon_id (str): Anonymized patient identifier.
    nifti_src_dir (Path): Path to the directory containing the NIfTI files.
    pat_dir (Path): Path to the patient's directory where NIfTI files should be copied.
    logger (Logger): Logger object for logging the process.
    cur (sqlite3.Cursor): Cursor object for executing SQL queries.
    conn (sqlite3.Connection): Connection object for connecting to the database.
    tablename (str): Name of the database table.
    """
    for study_dir in (nifti_src_dir / anon_id).iterdir():
        target_study_dir = pat_dir / 'niftis' / study_dir.name
        if not target_study_dir.exists():
            logger.info(f'Creating directory: {target_study_dir}')
            target_study_dir.mkdir(parents=True, exist_ok=True)
            for nifti_fpath in study_dir.iterdir():
                if 'ep_' in nifti_fpath.name or 'tse' in nifti_fpath.name or 'roi' in nifti_fpath.name or 'ROI' in nifti_fpath.name:
                    cmd = [
                        'cp',
                        '-r',
                        str(nifti_fpath),
                        str(target_study_dir)
                    ]
                    try:
                        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
                        logger.info(f'Copied {nifti_fpath} to {target_study_dir}')
                    except subprocess.CalledProcessError as e:
                        logger.error(f'Failed to copy {nifti_fpath} to {target_study_dir}: {e}')
        else:
            logger.info(f'\tStudy directory already exists: {target_study_dir}')
        
    # Check if there are any study directories in the patient's DICOM directory
    study_dirs = list((pat_dir / 'niftis').iterdir())
    if any(f.is_dir() for f in study_dirs):
        try:
            # Begin a transaction
            conn.execute('BEGIN')
            
            # Update the database to indicate that the niftis have been copied
            cur.execute(f"""
                UPDATE {tablename}
                SET has_niftis = 1
                WHERE anon_id = ?
            """, (anon_id,))
            
            # Commit the transaction
            conn.commit()
            logger.info(f"\tDatabase updated for patient {anon_id} to indicate niftis have been copied.")
        except sqlite3.Error as e:
            # Rollback the transaction on error
            conn.rollback()
            logger.error(f"Failed to update database for patient {anon_id}: {e}")
    else:
        logger.warning(f"Patient {anon_id} has no nifti directories, this should be investigated.")


def copy_anonymized_dicoms_if_needed(
    anon_id: str,
    anon_dcms_dir: Path,
    pat_dir: Path,
    logger: logging.Logger,
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection,
    tablename: str = 'kspace_dset_info'
) -> None:
    """
    Copies DICOM files from the anonymized DICOMs directory to the patient's DICOM directory.
    And updates the database to indicate that the DICOMs have been copied.
    
    Parameters:
    anon_id (str): Anonymized patient identifier.
    anon_dcms_dir (Path): Path to the directory containing the anonymized DICOMs.
    pat_dir (Path): Path to the patient's directory where DICOMs should be copied.
    logger (Logger): Logger object for logging the process.
    cur (sqlite3.Cursor): Cursor object for executing SQL queries.
    conn (sqlite3.Connection): Connection object for connecting to the database.
    """
    for study_dir in (anon_dcms_dir / anon_id).iterdir():
        target_study_dir = pat_dir / 'dicoms' / study_dir.name
        if not target_study_dir.exists():
            logger.info(f'Creating directory: {target_study_dir}')
            target_study_dir.mkdir(parents=True, exist_ok=True)
            for seq_dir in study_dir.iterdir():
                if 'ep_' in seq_dir.name or 'tse' in seq_dir.name:
                    cmd = [
                        'cp',
                        '-r',
                        str(seq_dir),
                        str(target_study_dir)
                    ]
                    try:
                        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
                        logger.info(f'Copied {seq_dir} to {target_study_dir}')
                    except subprocess.CalledProcessError as e:
                        logger.error(f'Failed to copy {seq_dir} to {target_study_dir}: {e}')
        else:
            logger.info(f'\tStudy directory already exists: {target_study_dir}')
            # Optionally, log more details or handle existing directory case
    
    # Check if there are any study directories in the patient's DICOM directory
    study_dirs = list((pat_dir / 'dicoms').iterdir())
    if any(f.is_dir() for f in study_dirs):
        try:
            # Begin a transaction
            conn.execute('BEGIN')
            
            # Update the database to indicate that the DICOMs have been copied
            cur.execute(f"""
                UPDATE {tablename}
                SET has_all_dicoms = 1
                WHERE anon_id = ?
            """, (anon_id,))
            
            # Commit the transaction
            conn.commit()
            logger.info(f"\tDatabase updated for patient {anon_id} to indicate DICOMs have been copied.")
        except sqlite3.Error as e:
            # Rollback the transaction on error
            conn.rollback()
            logger.error(f"Failed to update database for patient {anon_id}: {e}")
    else:
        logger.warning(f"Patient {anon_id} has no DICOM directories, this should be investigated.")


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


def format_date(date_string: str):
    date_object = datetime.strptime(date_string, "%Y%m%d")
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date


def extract_t2_tra_metadata(
    pat_dir: Path,
    study_date: str,
    logger: logging.Logger = None,
    cur: sqlite3.Cursor = None,
    tablename: str = 'dicom_headers_v1'
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
    seq_id, anon_id = pat_dir.name.split('_')
    dicoms_dir = Path(pat_dir / 'dicoms' / study_date)
    
    sd = study_date.replace('-', '')
    
    # Execute SQL query
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
    
    try:
        first_dcm_file = next(dicom_t2_dir.glob('*'))
    except StopIteration:
        msg = f"DICOM file not found in {dicom_t2_dir}"
        logger.error(msg) if logger else print(msg)
        raise FileNotFoundError(msg)

    # Read the DICOM file and extract metadata
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
    
    if logger:
        logger.info(f"\tExtracted DICOM metadata: {info_dict}")

    return info_dict

                
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
    
    
def perform_conversion(pat_dir,
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


def check_mri_date_exists(cur: sqlite3.Cursor, seq_id: str, tablename: str = 'kspace_dset_info') -> bool:
    """
    Check if the MRI date for a specific patient scan is already set in the database.

    Parameters:
    cur (sqlite3.Cursor): Database cursor.
    seq_id (str): The sequence ID of the patient.

    Returns:
    bool: True if the MRI date exists and is not null, False otherwise.
    """
    cur.execute(f"SELECT mri_date FROM {tablename} WHERE seq_id = ?", (seq_id,))
    result = cur.fetchone()
    return result is not None and result[0] is not None


def get_study_date_matching_kspace_date(
    raw_kspace_rootdir_all_patients: Path,
    seq_id: str,
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection,
    logger: logging.Logger,
    tablename: str = 'kspace_dset_info'
) -> str:
    """
    Retrieves and updates the study date for a patient based on the sequence ID.

    This function searches for the first DICOM file in the patient's directory, extracts the study date, and updates 
    the corresponding database record. It returns the study date for further use.

    Parameters:
    raw_kspace_rootdir_all_patients (Path): The root directory containing patient k-space data.
    seq_id (str): The sequence ID of the patient.
    cur (sqlite3.Cursor): Database cursor.
    conn (sqlite3.Connection): Database connection.
    logger (logging.Logger): Logger for logging the process.

    Returns:
    str: The study date extracted from the DICOM file.

    Raises:
    ValueError: If no DICOM file is found or the study date is not in the DICOM file.
    """
    unanon_dcm_dir = raw_kspace_rootdir_all_patients / f"{seq_id}_patient_umcg_done" / 'dicoms'
    study_date = None
    
    for dcm_file in unanon_dcm_dir.glob('**/*.IMA'):
        try:
            study_date = pydicom.dcmread(dcm_file).StudyDate
            if study_date:
                break
        except Exception as e:
            logger.error(f"Error reading DICOM file {dcm_file}: {e}")
            continue

    if study_date:
        # Update mri date where seq_id 
        cur.execute(f"UPDATE {tablename} SET mri_date = ? WHERE seq_id = ?", (study_date, seq_id))
        conn.commit()
        logger.info(f"Updated study date in database for patient {seq_id} to {study_date}.")
    else:
        error_msg = f"Study date not found for patient {seq_id}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return study_date


def get_study_date(cur: sqlite3.Cursor, tablename: str, seq_id: str) -> str:
    """
    Retrieves the study date for a patient based on the sequence ID.
    
    Parameters:
    cur (sqlite3.Cursor): Database cursor.
    tablename (str): Name of the table in the database.
    seq_id (str): The sequence ID of the patient.
    
    Returns:
    str: The study date extracted from the database.
    """
    query = f"SELECT mri_date FROM {tablename} WHERE seq_id = ?"
    study_date = cur.execute(query, (seq_id,)).fetchone()[0]
    return str(study_date)

    
def transfer_pat_data_to_habrok(
    anon_id: str,
    pat_dir: Path,
    logger: logging.Logger,
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection
) -> None:
    """
    Transfers the specified patient's data to the Habrok server using rsync,
    excluding the 'mrds' directory. Checks the database to ensure all required
    data conditions are met before initiating the transfer. Updates the database
    upon successful transfer.

    Parameters:
    anon_id (str): Anonymized ID of the patient.
    pat_dir (Path): Path to the patient's data directory.
    logger (logging.Logger): Logger for logging messages.
    cur (sqlite3.Cursor): Database cursor for executing SQL queries.
    conn (sqlite3.Connection): Database connection object.

    Raises:
    Exception: If an error occurs during data transfer or database operations.
    """
    username    = 'p290820'
    servername  = 'interactive1.hb.hpc.rug.nl'
    destination = '/scratch/p290820/datasets/003_umcg_pst_ksps/pat_data_full'
    habrok_dest = f"{username}@{servername}:{destination}"  
    exclude_dir = "--exclude=mrds"  # Excludes the 'mrds' directory

    try:
        # Check if all required data is present
        cur.execute("""
            SELECT has_all_dicoms, has_h5, has_niftis
            FROM kspace_dset_info
            WHERE anon_id = ?
        """, (anon_id,))
        result = cur.fetchone()
        if result and all(int(field) == 1 for field in result):
            # Execute rsync command
            rsync_command = ["rsync", "-avz", exclude_dir, str(pat_dir), habrok_dest]
            subprocess.run(rsync_command, check=True)

            # Update database upon successful transfer
            cur.execute("""
                UPDATE kspace_dset_info
                SET transfered_to_habrok = 1
                WHERE anon_id = ?
            """, (anon_id,))
            conn.commit()

            logger.info(f"Successfully transferred patient data for {anon_id} to Habrok and updated the database.")
        else:
            logger.warning(f"Patient {anon_id} does not have all required data for transfer. Missing fields: {result}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to transfer patient data for {anon_id}. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while processing patient {anon_id}. Error: {e}")
        raise
    

def main():
    args = parse_args(verbose=True)
    
    paths = {
        'umcglib_root': Path(args.umcglib_dir),
        'ksp_in_dir': Path(args.fastmri1_ksp_fpath),
        'out_dir': Path(args.fastmri2_out_fpath),
        'anon_dcms_dir': Path(args.anon_dcms_dir),
        'nifti_dir': Path(args.nifti_dir),
        'mrd_xml': Path(args.umcglib_dir) / 'src' / 'umcglib' / 'kspace' / 'custom_param_maps' / 'IsmrmrdParameterMap_Siemens_DIAG_UMCG.xml',
        'mrd_xsl': Path(args.umcglib_dir) / 'src' / 'umcglib' / 'kspace' / 'custom_param_maps' / 'IsmrmrdParameterMap_Siemens_DIAG_UMCG.xsl',
        'pat_data_dir': Path(args.fastmri2_out_fpath) / 'data' / 'pat_data'
    }
    
    # These patients are excluded for h5 conversion for now for some reason
    exclusion_dict = {
        '0050': 'ANON8824369', # Reason: study date not found in DB. Therefore no nifti path. Also Recidief patient.
        '0065': 'ANON7467725', # Reason: Build_kspace_array_from_mrd_umcg() fails. ValueError: could not broadcast input array from shape (20,640) into shape (20,320), also a recidief mention
        '0117': 'ANON9692714', # Reason: Build_kspace_array_from_mrd_umcg() fails. ValueError: could not broadcast input array from shape (20,640) into shape (20,320), also a recidief mention
        '0130': 'ANON9827881', # Reason: Build_kspace_array_from_mrd_umcg() fails. ValueError: could not broadcast input array from shape (20,640) into shape (20,320), also a recidief mention
    }
    
    logger = setup_logger(paths['out_dir']/'logs', use_time=False)
    conn   = connect_to_database(paths['out_dir']/'database'/'dbs'/'master_habrok_20231106.db')
    table  = 'kspace_dset_info'
    cur    = conn.cursor()

    # Create the dir structure for the processed data
    create_and_log_patient_dirs(paths['ksp_in_dir'], paths['pat_data_dir'], args.debug, logger)

    # Update get the patient info and update the database
    patients = get_patient_info(paths['pat_data_dir'], logger, verbose=False)
    
    if False:       # Temporary to write the patient info to a csv file
        write_pat_info_to_file(patients, logger, paths['out_dir'])
    
    # loop over the patients and update the database with the patient info and anonymize the .dat files
    for pat_idx, (seq_id, anon_id, pat_dir) in enumerate(patients):
        logger.info(f"Processing patient {pat_idx + 1} of {len(patients)}")

        # Skip patients that are in the exclusion list
        if seq_id in exclusion_dict.keys():
            logger.info(f"\t SKIPPING patient {seq_id} because it is in the exclusion list.")
            continue
        
        update_patient_info_in_db(cur, conn, table, seq_id, anon_id, pat_dir, logger)

        # Convert/Anonymize .dat files to .mrd  if it has not happened already
        raw_ksp_dir = paths['ksp_in_dir'] / f"{(pat_dir.name).split('_')[0]}_patient_umcg_done" / 'kspaces'
        process_mrd_if_needed(cur, pat_dir, raw_ksp_dir, logger, paths['mrd_xml'], paths['mrd_xsl'], table)

        # Copy the dicoms and nifti files from the anonymized dicoms and nifti dirs to the patient dicoms dir
        copy_anonymized_dicoms_if_needed(anon_id, paths['anon_dcms_dir'], pat_dir, logger, cur, conn)
        copy_nifti_files_if_needed(anon_id, paths['nifti_dir'], pat_dir, logger, cur, conn)
        
        # Check if the MRI date is already set in the database
        if not check_mri_date_exists(cur, seq_id):
            study_date = get_study_date_matching_kspace_date(paths['ksp_in_dir'], seq_id, cur, conn, logger)
        else:
            study_date = get_study_date(cur, table, seq_id)
            logger.info(f"\tMRI date for patient {seq_id} already exists in database.")
        study_date = format_date(study_date)
        
        # Convert the .mrd files into .h5 files
        convert_mrd_to_h5(
            pat_dir        = Path(pat_dir),
            study_date     = study_date,
            logger         = logger,
            cur            = cur,
            conn           = conn,
        )
        
        # Transfer all relevant files to Habrok
        if args.transfer:
            transfer_pat_data_to_habrok(anon_id, pat_dir, logger, cur, conn)

    conn.close()

#########################################################################
if __name__ == "__main__":
    main()