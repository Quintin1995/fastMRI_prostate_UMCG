# Use case
Convert raw multi-average multi-coil k-space (.dat files) into .h5 files that can be used for training and inference of deep learning reconstruction models.


# General Pipeline Processing Steps
1. The database is checked to see which steps must still be performed for the current patient.
2. Convert .dat to .mrd.
    * For this we use the 'siemens_to_ismrmrd' tool [Siemens to ISMRMRD Github](https://github.com/ismrmrd/siemens_to_ismrmrd).
3. Copy anonymized dicom if required (T2w).
    * The FOV of the k-space is differnt than the DICOM. Therefore we store the DICOM and its headers so that we can restore the 'correct' FOV after reconstruction.
    * This FOV can be used for the lesion segmentations if they are available.
4. Copy niftis if required.
5. Link the dicom and k-space based on the studydate
6. Convert .mrd to .h5.
    1. Convert to array
        1. Build k-space array from ISMRMRD object
            1. Get first acquisition.
            2. Initialize empty matrix.
            3. Iteratively fill the array based on the location within the object.
        2. Reorder k-space array (odd and even lines). Fix interleaving.
        3. Remove zero-padding if required
        4. Crop k-space in phase encoding direction if required.
        5. Normalize k-space to a reference k-space if required.
        6. Root Sum of squares (RSS) for visualization of correct processing if required.
    2. Extract T2w Tra meta data to be stored in the H5
7. Transfer to a high performance cluster for Training or Inference of a Deep Learning Reconstruction model.


# Configuration
There is a configuration file.

# K-space averages
The first and third averages masure odd lines, while the second average measures even lines. An option to create 1 k-space from all three averages:
K~full~ = (K~1~+K~3~)/2 + K~2~
![avg_example](figures/average_combination_example.png)


# Database
Since the processing of the data from .dat to .mrd to .h5 takes multiple steps and numerous operations, we keep track of the status of each step in database table in SQLite.
Configurations for this can be found in the configuration file.


# Examples
Example slice of k-space averages and their respective ordering for NYU and UMCG data.
![example](figures/kspace_example_nyu_and_umcg.png)