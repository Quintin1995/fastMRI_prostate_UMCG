from datetime import datetime
import os
import glob
from typing import List, Dict
import fnmatch
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pydicom import dcmread


KEY = '7531598426'


def format_date(date_string: str):
    date_object = datetime.strptime(date_string, "%Y%m%d")
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date


def create_directory_structure(base_path: Path):
    sub_dirs = ['lesion_bbs', 'recons', 'dicoms', 'mrds', 'h5s', 'rois', 'niftis']

    for sub_dir in sub_dirs:
        (base_path / sub_dir).mkdir(parents=True, exist_ok=True)


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


def get_dicom_headers(fpath: str) -> object:
    '''
    Description:
        This function returns the dicom headers of a dicom file.
    Args:
        fpath (str): The path to the dicom file.
    Returns:
        ds (object): The dicom headers.
    '''
    hdrs_object = dcmread(fpath)
    return hdrs_object


def pretty_print_xml(xml_element):
    """
    Returns a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(xml_element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


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


def print_dict_values(d, depth=0):
    '''
    Print the values of a dictionary with a given depth
    '''
    for key, value in d.items():
        if isinstance(value, dict):
            print_dict_values(value, depth + 1)
        else:
            print(f"\t" * depth + f"{key}: {value}")


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


# def decipher(anon_id: str, key: str):
#     '''
#     Description:
#         Deciphers the anonymized patient id to the unanonymized patient id
#     Arguments:
#         - anon_id: anonymized patient id
#         - key: key to use for deciphering
#     Returns:
#         - unanon_pat_id: unanonymized patient id
#     '''

#     output = []
#     non_digit_char_count = 0

#     for c_idx, char in enumerate(anon_id[4:]):
#         if not str.isdigit(char):
#             output.append(char)
#             non_digit_char_count += 1
#             continue
#         key_idx = c_idx - non_digit_char_count
#         k = int(key[key_idx])
#         out_char = (int(char) - k) % 10
#         output.append(str(out_char))
    
#     return ''.join(output)


# def encipher(patient_id: str, key: str):
#     '''
#     Description:
#         Enciphers the patient id to the anonymized patient id
#     Arguments:
#         - patient_id: patient id
#         - key: key to use for enciphering
#     Returns:
#         - anon_pat_id: anonymized patient id
#     '''
#     output = []
#     non_digit_char_count = 0

#     for c_idx, char in enumerate(patient_id):
#         if not str.isdigit(char):
#             output.append(char)
#             non_digit_char_count += 1
#             continue
#         key_idx = c_idx - non_digit_char_count
#         k = int(key[key_idx])
#         out_char = (int(char) + k) % 10
#         output.append(str(out_char))
    
#     return f"ANON{''.join(output)}"


def dt_str():
    now = datetime.now()
    return now.strftime("%m%d%H%M")


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