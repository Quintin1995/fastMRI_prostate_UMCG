import ismrmrd
# from assets.writers import save_crop_to_file, save_crops_to_file
import numpy as np
import logging
import time


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


def echo_train_length(dset, verbose=False) -> int:
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
    if verbose:
        start_time = time.time()
        print(f"Computing Echo Train Length...")

    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n).isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            firstacq = n
            break
    for n in range(dset.number_of_acquisitions()):
        if dset.read_acquisition(n)._head.idx.slice != dset.read_acquisition(0)._head.idx.slice:
            if verbose:
                print(f"\tFound different slice at acquisition {n}, so echo train length is {n - firstacq}. Time elapsed in seconds: {time.time() - start_time}")
            return n - firstacq
    raise Exception("Couldn't find different slices in the dataset")


def echo_train_count(dset, echo_train_len=25, verbose=False) -> int:
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
    if verbose:
        start_time = time.time()
        print(f"Computing Echo Train Count... Going to loop through all acquisitions, until we find the second average. Then ETL = int(count / (nslices * ETL))")

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

    if verbose:
        print(f"\tFound second average at acquisition {n}, so echo train count is {count}. Time elapsed in seconds: {time.time() - start_time}")
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


# def view_stats_kspace(kspace: np.ndarray, tmpdir: str, pat_num: str, save_crops: False) -> None:
#     '''
#     Arguments:
#         - kspace: numpy array of kspace data in shape (navgs, nslices, ncoils, rNx, eNy + 1) complex
#         - tmpdir: directory to save the crops to
#         - pat_num: patient number
#         - save_crops: whether to save crops of the first two slices
#     '''
#     print(f"kspace shape: {kspace.shape}")
#     print(f"kspace dtype: {kspace.dtype}")

#     if save_crops:
#         save_crop_to_file(tmpdir, pat_num, kspace, cropsize=30, avg=0, slice=0, coil=0)
#         save_crop_to_file(tmpdir, pat_num, kspace, cropsize=30, avg=1, slice=0, coil=0)
#         save_crop_to_file(tmpdir, pat_num, kspace, cropsize=30, avg=2, slice=0, coil=0)
#         save_crops_to_file(tmpdir, pat_num, kspace, cropsize=30, slice=0, coil=0)
        

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