import sqlite3
import logging
import subprocess
import h5py
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import ismrmrd

from assets.util import print_element, find_t2_tra_kspace_files, print_headers
from assets.util import convert_dict_to_xml, pretty_print_xml


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
    

# def process_mrd_if_needed(
#     cur: sqlite3.Cursor,
#     pat_out_dir: Path,
#     raw_ksp_dir: Path,
#     mrd_xml: Path,
#     mrd_xsl: Path,
#     tablename: str,
#     logger: logging.Logger,
# ) -> None:
#     """
#     Perform .dat to .mrd anonymization if not done already.

#     Parameters:
#     cur: SQLite cursor object.
#     pat_out_dir: Path object for patient output directory.
#     raw_ksp_dir: Path object for raw k-space directory.
#     logger: Logger object for logging messages.
#     mrd_xml: Path object for MRD XML file.
#     mrd_xsl: Path object for MRD XSL file.
#     tablename (str): Name of the table in the database.
#     """
#     cur.execute(f"SELECT has_mrds FROM {tablename} WHERE data_dir = ?", (str(pat_out_dir),))
#     has_mrd = cur.fetchone()[0]

#     if has_mrd is None or has_mrd == 0:
#         success = anonymize_mrd_files(pat_out_dir, raw_ksp_dir, logger, mrd_xml, mrd_xsl)  # Implement this function

#         if success:
#             cur.execute(f"""
#                 UPDATE {tablename}
#                 SET has_mrds = 1
#                 WHERE data_dir = ?
#             """, (str(pat_out_dir),))


def process_mrd_if_needed(
    conn: sqlite3.Connection,  # Pass the connection object instead of the cursor
    pat_out_dir: Path,
    raw_ksp_dir: Path,
    mrd_xml: Path,
    mrd_xsl: Path,
    tablename: str,
    logger: logging.Logger,
) -> None:
    """
    Perform .dat to .mrd anonymization if not done already.

    Parameters:
    conn: SQLite connection object.
    pat_out_dir: Path object for patient output directory.
    raw_ksp_dir: Path object for raw k-space directory.
    logger: Logger object for logging messages.
    mrd_xml: Path object for MRD XML file.
    mrd_xsl: Path object for MRD XSL file.
    tablename (str): Name of the table in the database.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT has_mrds FROM {tablename} WHERE data_dir = ?", (str(pat_out_dir),))
    has_mrd = cur.fetchone()[0]

    if has_mrd is None or has_mrd == 0:
        success = anonymize_mrd_files(pat_out_dir, raw_ksp_dir, logger, mrd_xml, mrd_xsl)  # Implement this function

        if success:
            try:
                with conn:  # This will manage the transaction
                    cur.execute(f"""
                        UPDATE {tablename}
                        SET has_mrds = 1
                        WHERE data_dir = ?
                    """, (str(pat_out_dir),))
                logger.info(f"MRD files for {pat_out_dir} have been processed and database updated.")
            except sqlite3.Error as e:
                logger.error(f"Failed to update database for {pat_out_dir}: {e}")
 
            
            

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