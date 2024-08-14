import argparse
import logging
from pathlib import Path
import sqlite3
from typing import Tuple, List
import subprocess
import pydicom
import yaml

from assets.util import create_directory_structure, setup_logger, format_date
from assets.cipher import encipher
from assets.operations_sqlite_db import get_study_date, connect_to_database, update_patient_info_in_db, check_mri_date_exists, transfer_pat_data_to_hpc
from assets.convert_to_mrd import process_mrd_if_needed
from assets.convert_to_h5 import convert_mrd_to_h5
from assets.writers import write_pat_info_to_file


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
    
    # Path to the config file to run the program
    parser.add_argument("-cfg",
                        "--config_fpath",
                        type=str,
                        default="configs/config.yaml",         # this is the FASTMRI1 drive, EXTERNAL SSD (D:)
                        help="Path to the config file.")
    
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


def create_and_log_patient_dirs(
    ksp_dir: Path,
    out_dir: Path,
    logger: logging.Logger = None,
    debug=False,
    key=None,
    seq_id_filter: List[str] = None,
    **kwargs,
) -> None:
    """
    Process patient directories to create the required directory structure and
        log relevant information.
    Parameters:
    - dataset_drive_locs (List[Path]): List of paths to the dataset directories.
    - out_dir (Path): The directory where the processed data will be stored.
    - logger (logging.Logger): Logger object for logging information.
    - debug (bool): Flag for debugging. If True, the loop breaks after 2 iterations.
    - key (str): Key for enciphering patient number.
    - seq_id_filter (List[str]): List of sequence IDs to filter the patients.
    """
    
    assert key is not None, "Key for enciphering patient number is not provided."
    
    pat_dirs = filter_and_sort_patient_dirs(ksp_dir)
    
    for pat_idx, pat_dir in enumerate(pat_dirs, start=1):
        seq_id = pat_dir.split('_')[0]
        
        if seq_id not in seq_id_filter:
            print(f"Skipping patient {seq_id} because it is not in the filter list.")
            continue
        
        pat_dir_path = ksp_dir / pat_dir
        dicom_dir_path = pat_dir_path / 'dicoms'

        assert len([f.name for f in dicom_dir_path.iterdir() if f.is_dir()]) == 1, \
            "There should be only one dir in the dicom dir"

        pat_dcm_dir = [f for f in dicom_dir_path.iterdir() if f.is_dir()][0]
        pat_num = pat_dcm_dir.name.split('_')[-1]
        anon_pat_id = encipher(pat_num, key=key)  # Assuming encipher is defined elsewhere

        new_pat_dir_path = out_dir / f"{seq_id}_{anon_pat_id}"

        if not new_pat_dir_path.exists():
            create_directory_structure(new_pat_dir_path)  # Assuming this function is defined elsewhere
            logger.info("Created new directory: %s", new_pat_dir_path)
        else:
            # logger.info("Directory already exists: %s", new_pat_dir_path)
            pass

        if debug and pat_idx > 2:
            break


def get_patient_info(
    target_dir: Path,
    logger: logging.Logger = None,
    seq_id_filter: List[str] = None,
    verbose: bool = False,
) -> Tuple[str, str, Path]:
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
    
    # only keep the patients that are in the filter list
    if seq_id_filter is not None:
        pat_info = [pat for pat in pat_info if pat[0] in seq_id_filter]

    if verbose:
        for pat_number, anon_id, full_path in pat_info:
            logger.info(f"Patient Number: {pat_number}, anon_id: {anon_id}, Full Path: {full_path}")

    return pat_info


def copy_nifti_files_if_needed(
    anon_id: str,
    nifti_src_dir: Path,
    pat_dir: Path,
    conn: sqlite3.Connection,
    logger: logging.Logger,
    tablename: str = None,     # This should be the kspace table
) -> None:
    """
    Copies NIfTI files from the source directory to the patient's NIfTI directory using the cp command.

    Parameters:
    anon_id (str): Anonymized patient identifier.
    nifti_src_dir (Path): Path to the directory containing the NIfTI files.
    pat_dir (Path): Path to the patient's directory where NIfTI files should be copied.
    logger (Logger): Logger object for logging the process.
    conn (sqlite3.Connection): Connection object for connecting to the database.
    tablename (str): Name of the table in the database. This should be the kspace table.
    """
    assert tablename is not None, "Table name is not provided."
    
    copied = False
    already_copied = False

    for study_dir in (nifti_src_dir / anon_id).iterdir():
        target_study_dir = pat_dir / 'niftis' / study_dir.name
        if not target_study_dir.exists():
            logger.info(f'Creating directory: {target_study_dir}')
            target_study_dir.mkdir(parents=True, exist_ok=True)
            for nifti_fpath in study_dir.iterdir():
                if 'ep_' in nifti_fpath.name or 'tse' in nifti_fpath.name or 'roi' in nifti_fpath.name or 'ROI' in nifti_fpath.name:
                    cmd = ['cp', '-r', str(nifti_fpath), str(target_study_dir)]
                    try:
                        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
                        logger.info(f'SUCCESS: Copied {nifti_fpath} to {target_study_dir}')
                        copied = True
                    except subprocess.CalledProcessError as e:
                        logger.error(f'FAILED to copy {nifti_fpath} to {target_study_dir}: {e}')
        else:
            already_copied = True
            logger.info(f'Study directory already exists: {target_study_dir}')

    if copied:
        try:
            with conn:
                conn.execute(f"""
                    UPDATE {tablename}
                    SET has_niftis = 1
                    WHERE anon_id = ?
                """, (anon_id,))
                logger.info(f"Database updated for patient {anon_id} to indicate niftis have been copied.")
        except sqlite3.Error as e:
            logger.error(f"Failed to update database for patient {anon_id}: {e}")
    else:
        if already_copied:
            logger.info(f"Nifti directories already copied for patient {anon_id}.")
        else:
            logger.warning(f"No nifti directories copied for patient {anon_id}. This should be investigated.")


def copy_anonymized_dicoms_if_needed(
    anon_id: str,
    anon_dcms_dir: Path,
    pat_dir: Path,
    conn: sqlite3.Connection,
    tablename: str, # This should be the kspace table
    logger: logging.Logger,
) -> None:
    """
    Copies DICOM files from the anonymized DICOMs directory to the patient's DICOM directory using the cp command.
    And updates the database to indicate that the DICOMs have been copied.

    Parameters:
    anon_id (str): Anonymized patient identifier.
    anon_dcms_dir (Path): Path to the directory containing the anonymized DICOMs.
    pat_dir (Path): Path to the patient's directory where DICOMs should be copied.
    logger (Logger): Logger object for logging the process.
    conn (sqlite3.Connection): Connection object for connecting to the database.
    tablename (str): Name of the table in the database.
    """
    assert tablename is not None, "Table name is not provided."
    
    copied = False
    already_copied = False

    for study_dir in (anon_dcms_dir / anon_id).iterdir():
        target_study_dir = pat_dir / 'dicoms' / study_dir.name
        if not target_study_dir.exists():
            logger.info(f'Creating directory: {target_study_dir}')
            target_study_dir.mkdir(parents=True, exist_ok=True)
            for seq_dir in study_dir.iterdir():
                if 'ep_' in seq_dir.name or 'tse' in seq_dir.name:
                    cmd = ['cp', '-r', str(seq_dir), str(target_study_dir)]
                    try:
                        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
                        logger.info(f'SUCCESS: Copied {seq_dir} to {target_study_dir}')
                        copied = True
                    except subprocess.CalledProcessError as e:
                        logger.error(f'FAILED to copy {seq_dir} to {target_study_dir}: {e}')
        else:
            already_copied = True
            logger.info(f'Study directory already exists: {target_study_dir}')

    if copied:
        try:
            with conn:
                conn.execute(f"""
                    UPDATE {tablename}
                    SET has_all_dicoms = 1
                    WHERE anon_id = ?
                """, (anon_id,))
                logger.info(f"Database updated for patient {anon_id} to indicate DICOMs have been copied.")
        except sqlite3.Error as e:
            logger.error(f"Failed to update database for patient {anon_id}: {e}")
    else:
        if already_copied:
            logger.info(f"DICOM directories already copied for patient {anon_id}.")
        else:
            logger.warning(f"No DICOM directories copied for patient {anon_id}. This should be investigated.")


def get_study_date_matching_kspace_date(
    raw_kspace_rootdir_all_patients: Path,
    seq_id: str,
    conn: sqlite3.Connection,
    logger: logging.Logger,
    tablename: str = None      # This should be the kspace table
) -> str:
    """
    Retrieves and updates the study date for a patient based on the sequence ID.
    
    This function searches for the first DICOM file in the patient's directory, extracts the study date, and updates 
    the corresponding database record. It returns the study date for further use.
    
    Parameters:
    raw_kspace_rootdir_all_patients (Path): The root directory containing patient k-space data.
    seq_id (str): The sequence ID of the patient.
    conn (sqlite3.Connection): Database connection.
    logger (logging.Logger): Logger for logging the process.
    tablename (str): Name of the table in the database. This should be the kspace table.
    
    Returns:
    str: The study date extracted from the DICOM file.
    
    Raises:
    ValueError: If no DICOM file is found or the study date is not in the DICOM file.
    """
    assert tablename is not None, "Table name is not provided."
    
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
        try:
            with conn:
                conn.execute(f"UPDATE {tablename} SET mri_date = ? WHERE seq_id = ?", (study_date, seq_id))
                logger.info(f"Updated study date in database for patient {seq_id} to {study_date}.")
        except sqlite3.Error as e:
            logger.error(f"Failed to update study date for patient {seq_id}: {e}")
            raise
    else:
        error_msg = f"Study date not found for patient {seq_id}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return study_date


def load_config():
    args = parse_args(verbose=True)
    with open(args.config_fpath, 'r', encoding='UTF-8') as file:
        cfg = yaml.safe_load(file)
    
    keys_to_make_a_path = ['ksp_in_dir', 'out_root', 'mrd_xml', 'mrd_xsl', 'anon_dcms_dir', 'nifti_dir', 'sqlite_db_fpath']
    for key in keys_to_make_a_path:
        cfg[key] = Path(cfg[key])
    
    for k, v in cfg.items():
        print(f"{k}:\t{v}  \tdtype: {type(v)}")
    print("")
    return cfg


def main():
    """
    MOUNTING AN EXTERNAL DRIVE ON WSL:
    
    sudo umount /mnt/d
    sudo mount -t drvfs D: /mnt/d
    sudo umount /mnt/f
    sudo mount -t drvfs F: /mnt/f
    
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
    cfg = load_config()

    # These patients are excluded for h5 conversion for now for some reason
    exclusion_dict = {
        '0050': 'ANON8824369', # Reason: study date not found in DB. Therefore no nifti path. Also Recidief patient.
        '0065': 'ANON7467725', # Reason: Build_kspace_array_from_mrd_umcg() fails. ValueError: could not broadcast input array from shape (20,640) into shape (20,320), also a recidief mention
        '0117': 'ANON9692714', # Reason: Build_kspace_array_from_mrd_umcg() fails. ValueError: could not broadcast input array from shape (20,640) into shape (20,320), also a recidief mention
        '0130': 'ANON9827881', # Reason: Build_kspace_array_from_mrd_umcg() fails. ValueError: could not broadcast input array from shape (20,640) into shape (20,320), also a recidief mention
    }
    
    logger = setup_logger(cfg['out_root']/'logs', use_time=False)
    conn   = connect_to_database(cfg['out_root']/cfg['sqlite_db_fpath'])

    # Create the dir structure for the processed data
    pat_data_dir = cfg['out_root'] / 'data' / 'pat_data'
    create_and_log_patient_dirs(cfg['ksp_in_dir'], pat_data_dir, logger, **cfg)

    # Get the patient info and update the database SQlite
    patients = get_patient_info(pat_data_dir, logger=logger, seq_id_filter=cfg['seq_id_filter'], verbose=False)
    
    if False:       # Temporary to write the patient info to a csv file
        write_pat_info_to_file(patients, logger, cfg['out_dir'], **cfg)
    
    # Iterate over patients, update  database with patient info and anonymize .dat files
    for pat_idx, (seq_id, anon_id, pat_dir) in enumerate(patients):
        logger.info(f"Processing patient {pat_idx + 1} of {len(patients)}")

        # Skip patients from the exclusion list
        if seq_id in exclusion_dict.keys():
            logger.info(f"\t SKIPPING patient {seq_id} because it is in the exclusion list.")
            continue
        update_patient_info_in_db(conn, seq_id, anon_id, pat_dir, logger, cfg['tablename_ksp'])

        # Convert/Anonymize .dat files to .mrd  if it has not happened already
        raw_ksp_dir = cfg['ksp_in_dir'] / f"{(pat_dir.name).split('_')[0]}_patient_umcg_done" / 'kspaces'
        process_mrd_if_needed(conn, pat_dir, raw_ksp_dir, cfg['mrd_xml'], cfg['mrd_xsl'], cfg['tablename_ksp'], logger)

        # Copy the dicoms and nifti files from the anonymized dicoms and nifti dirs to the patient dicoms dir
        copy_anonymized_dicoms_if_needed(anon_id, cfg['anon_dcms_dir'], pat_dir, conn, cfg['tablename_ksp'], logger)
        copy_nifti_files_if_needed(anon_id, cfg['nifti_dir'], pat_dir, conn, logger, cfg['tablename_ksp'])

        # Check if the MRI date is already set in the database
        if not check_mri_date_exists(conn, seq_id, cfg['tablename_ksp']):
            study_date = get_study_date_matching_kspace_date(cfg['ksp_in_dir'], seq_id, conn, logger, cfg['tablename_ksp'])
        else:
            study_date = get_study_date(conn, seq_id, cfg['tablename_ksp'])
            logger.info(f"\tMRI date for patient {seq_id} already exists in database.")
        study_date = format_date(study_date)
        
        convert_mrd_to_h5(
            pat_dir        = Path(pat_dir),
            study_date     = study_date,
            logger         = logger,
            conn           = conn,
            **cfg
        )
        
        # Transfer all relevant files to Habrok
        if cfg['transfer_to_hpc']:
            transfer_pat_data_to_hpc(anon_id, pat_dir, logger, conn, **cfg)

    conn.close()


#########################################################################
if __name__ == "__main__":
    main()