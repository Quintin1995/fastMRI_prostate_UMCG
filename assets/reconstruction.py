import numpy as np
from ismrmrdtools import transform
from pathlib import Path
import logging
from assets.writers import save_numpy_rss_as_nifti
from assets.operations_kspace import get_first_acquisition
import ismrmrd


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