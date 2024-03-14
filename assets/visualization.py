import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pydicom


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
    
    
