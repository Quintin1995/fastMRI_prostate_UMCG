# T2w Transversal/axial k-space Data Conversion for AI
This pipeline is designed for converting T2w Transversal/axial k-space data from Siemens MRI scanners into a format suitable for deep learning applications. The process involves three key stages: 

1. **Conversion Workflow**: 
   - Convert raw .dat files to .mrd format using the ISMRMRD framework.
   - Further convert .mrd files to .h5 format, maintaining all essential metadata and headers.
   
2. **Handling Interleaved Averages**:
   - The k-space data contains three interleaved averages. Unlike traditional methods that average these, this pipeline preserves all three averages for enhanced flexibility in reconstruction.

3. **Metadata Integrity**:
   - Headers and DICOM metadata are preserved to ensure accurate reconstruction, with an option to restore original Field of View (FOV) if needed yet to be released.

The output .h5 files are optimized for training and inference in AI models, with careful handling of k-space data to maintain high fidelity in the final reconstructed images.


# General Pipeline Processing Steps
1. **Check Database**: Determine which steps are required for the current patient.
2. **Convert .dat to .mrd**:
   - Utilize the [siemens_to_ismrmrd](https://github.com/ismrmrd/siemens_to_ismrmrd) tool.
3. **Anonymize DICOM**:
   - Store DICOM files and headers to restore the 'correct' FOV after reconstruction.
   - FOV used for lesion segmentation if available.
4. **Copy Niftis**: If required.
5. **Link DICOM and k-space**: Based on study date.
6. **Convert .mrd to .h5**:
   1. **Convert to Array**:
      - Build k-space array from ISMRMRD object.
      - Reorder k-space array and fix interleaving.
      - Remove zero-padding, crop, normalize, and apply Root Sum of Squares (RSS) if required.
   2. **Extract Metadata**: T2w Tra metadata is stored in the H5.
7. **Transfer Data**: Send to a high-performance cluster for deep learning reconstruction model training or inference.

# Configuration
*To Be Determined (TBD)*

# K-space Averages
The first and third averages measure odd lines, while the second average measures even lines. You can create one k-space from all three averages:
K<sub>full</sub> = (K<sub>1</sub>+K<sub>3</sub>)/2 + K<sub>2</sub>
![Average Example](figures/average_combination_example.png)

# Database
A database table in SQLite tracks the status of each processing step from .dat to .mrd to .h5. Configurations are stored in a configuration file.

# Examples
Example slice of k-space averages and their ordering for NYU and UMCG data.
![Example](figures/kspace_example_nyu_and_umcg.png)