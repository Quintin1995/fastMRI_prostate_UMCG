import argparse
import logging
from pathlib import Path
import sqlite3
from typing import Tuple, Dict
import subprocess
import pydicom

from assets.util import KEY, create_directory_structure, setup_logger, format_date
from assets.cipher import encipher
from assets.operations_sqlite_db import connect_to_database, update_patient_info_in_db, check_mri_date_exists
from assets.convert_to_mrd import process_mrd_if_needed
from assets.convert_to_h5 import convert_mrd_to_h5


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
    """
    MOUNTING AN EXTERNAL DRIVE ON WSL:
        This code reads from an external hard-drive (SSD) because we are dealing with a lot of data.
        In fact even from SSD (D) to another SSD (E)
        I use WSL on windows and it might not recongnize the SSDs as mounted drives.
        If you want to make sure that the mounted drive can be found:
        - Open the WSL terminal
        - Type 'cd /mnt'
        - Type 'ls'
        - You should see the mounted drives there.
        - if it doesnt exist, you can make the drive letter available by typing 'mkdir /mnt/e' for example.
        - Then the letter can be assigned to the drive that is inserted into the computer like:
        - 'sudo mount -t drvfs E: /mnt/e'
    """
    
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