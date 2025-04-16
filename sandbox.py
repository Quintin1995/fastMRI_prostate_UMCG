from pathlib import Path
import ismrmrd

local_dir = Path(r"\\zkh\appdata\research\Radiology\fastMRI_PCa\03_data\002_processed_pst_ksp_umcg_backup\data\pat_data")
mrd_fpath = local_dir / "0003_ANON5046358" / "mrds" / "meas_MID00401_FID373323_t2_tse_traobl_p2_384-out_2.mrd"

print(f"The fpath {mrd_fpath}")
print("File exists") if mrd_fpath.is_file() else print("File does not exist")

dset   = ismrmrd.Dataset(mrd_fpath, create_if_needed=False)
header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc    = header.encoding[0]
print(f"Successfully opened dataset: {mrd_fpath}")

ncoils     = header.acquisitionSystemInformation.receiverChannels
nslices    = enc.encodingLimits.slice.maximum + 1 if enc.encodingLimits.slice is not None else 1
eNy        = enc.encodedSpace.matrixSize.y
rNx        = enc.reconSpace.matrixSize.x
eTL        = 25 if DEBUG else echo_train_length(dset, verbose=True)                           # echo train length = 25
# eTC        = 13 if DEBUG else echo_train_count(dset, echo_train_len=eTL, verbose=True)       # echo train count = 11
firstacq   = get_first_acquisition(dset)
navgs      = 3 #if DEBUG else get_num_averages(firstacq=firstacq, dset=dset)
total_acqs = dset.number_of_acquisitions()
logger.info(f"\t navgs: {navgs}, nslices: {nslices}, ncoils: {ncoils}, rNx: {rNx}, eNy: {eNy}, first_qc: {firstacq}, total_acqs: {total_acqs}")

# # Loop through the rest of the acquisitions and fill the data array with the kspace data
# echo_train_mapping = {}
# init_col_idx = -1
# for acq_idx in range(firstacq, dset.number_of_acquisitions()):
#     acq          = dset.read_acquisition(acq_idx)
#     slice_idx    = acq.idx.slice
#     col_idx      = acq.idx.kspace_encode_step_1
#     avg_idx      = acq._head.idx.average
    
#     if init_col_idx == -1:
#         init_col_idx = col_idx
#     elif init_col_idx == col_idx:
#         print(f"Skipping acquisition {acq_idx} because col_idx is the same as init_col_idx", end="",)
#         continue

#     if acq_idx > 1000:
#         break