"""
Module: echo_train_mapping_exporter.py
Description:
  Processes .dat files using twixtools to extract image acquisition metadata,
  computes echo train mapping (if needed), and exports the ordering information
  into a SQLite database.
"""

from pathlib import Path
import sqlite3
import pandas as pd
import logging
import twixtools
import ismrmrd
from assets.operations_kspace import get_first_acquisition, echo_train_count, echo_train_length, get_num_averages
from typing import Dict, List
from datetime import datetime


def get_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a logger that writes to both console and a dated file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    logfile = log_dir / f"{name}_{date}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid adding handlers multiple times
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def create_db_table_if_not_exists(db_path: Path, table_name: str):
    """
    Create the specified SQLite table if it doesn't already exist.

    Parameters:
      db_path (Path): The path to the SQLite database file.
      table_name (str): The name of the table to create.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                aqc_idx          INTEGER,
                id               TEXT,
                fname            TEXT,
                avg_idx          INTEGER,
                col_idx          INTEGER,
                slice_idx        INTEGER,
                echo_train_idx   INTEGER,
                inner_et_counter INTEGER,
            );
        """)
        conn.commit()
        LOGGER.info(f"Table '{table_name}' ensured in database '{db_path}'.")
    except Exception as e:
        LOGGER.info(f"Error creating table: {e}")
    finally:
        conn.close()


def insert_acquisition_data_to_db(acq_data: list, db_path: Path, table_name: str):
    """
    Insert the acquisition data (list of dictionaries) into the SQLite database.
    """
    try:
        df = pd.DataFrame(acq_data)
        if 'id' not in df.columns:
            df['id'] = ""
        conn = sqlite3.connect(str(db_path))
        df.to_sql(table_name, con=conn, if_exists='append', index=False)
        conn.commit()
        LOGGER.info(f"Inserted {len(df)} rows into table '{table_name}'.")
    except Exception as e:
        LOGGER.info(f"Error inserting data into table: {e}")
    finally:
        conn.close()


def find_mrd_file_wrapper(id: str, mrd_root_dir: Path) -> Path:
    try:
        # Determine the MRD file path (implement your own find_mrd_fpath)
        mrd_pat_dir = mrd_root_dir / id / 'mrds'
        mrd_fpath = find_mrd_fpath(id, mrd_pat_dir)  # <-- Your helper function to locate the file
    except Exception as e:
        LOGGER.warning(f"Error finding MRD file for patient {id}: {e}")
        return []
    return mrd_fpath


def find_starting_acq_idx_first_echo_train(dset: ismrmrd.Dataset, firstacq: int) -> int:
    """
    Find the beginning of the first real echo train in the dataset.
    This is done by checking the column index of the first acquisition and skipping
    all acquisitions with the same column index until we find a different one.
    """
    init_col_idx = -1
    for acq_idx in range(firstacq, dset.number_of_acquisitions()):
        acq            = dset.read_acquisition(acq_idx)
        col_idx        = acq.idx.kspace_encode_step_1

        if init_col_idx == -1:  # first acquisition 
            init_col_idx = col_idx
        elif init_col_idx == col_idx:   # if the column index is the same as the initial one, we skip it
            LOGGER.info(f"Skipping acquisition {acq_idx} because col_idx is the same as init_col_idx") if acq_idx % 250 == 0 else None
            continue
        else:   # now we have started and skipped all the initial acquisitions
            LOGGER.info(f"Found starting acquisition index: {acq_idx} (col_idx: {col_idx})")
            return acq_idx


def process_patient_mrd(pat_id: str, cfg: Dict[str, str]):
    """
    Process a single patient: extract image acquisition metadata from the last
    measurement (assumed to be image data) and write it to the database.
    """
    LOGGER.info(f"\nProcessing patient: {pat_id}")
    mrd_fpath = find_mrd_file_wrapper(pat_id, cfg['mrd_root_dir'])

    try:
        dset = ismrmrd.Dataset(mrd_fpath, create_if_needed=False)
        LOGGER.info(f"Successfully loaded MRD data for patient {pat_id}.")
    except Exception as e:
        LOGGER.warning(f"Error loading MRD data for patient {pat_id}: {e}")
        return []

    firstacq   = get_first_acquisition(dset)
    eTL        = 25 #if DEBUG else echo_train_length(dset, verbose=True)                           # echo train length = 25
    echo_train_mapping = []

    last_avg         = 0
    last_slice       = 0
    inner_et_counter = 0
    last_et          = 0
    start            = find_starting_acq_idx_first_echo_train(dset, firstacq=firstacq)
    for acq_idx in range(start, dset.number_of_acquisitions()):
        # percentage of progress LOGGER.info
        LOGGER.info(f"Processing acquisition {acq_idx} of {dset.number_of_acquisitions()} ({(acq_idx / dset.number_of_acquisitions()) * 100:.2f}%)") if acq_idx % 250 == 0 else None
        acq            = dset.read_acquisition(acq_idx)
        slice_idx      = acq.idx.slice
        col_idx        = acq.idx.kspace_encode_step_1
        avg_idx        = acq._head.idx.average

        # we record the echo train lenght = 25, so we need to reset the counter if it is bigger than that
        if inner_et_counter >= eTL:
            inner_et_counter = 0

        # if the current slice index is bigger then the last recorded one, then we increment the echo trains counter. Then we have had the last slice and move on to the next echo train.
        if slice_idx < last_slice:
            LOGGER.info(f"acq_idx: {acq_idx}, avg_idx: {avg_idx}, col_idx: {col_idx}, cSlc: {slice_idx}, echo_train_idx: {last_et}")
            last_et += 1

        # if the current average is bigger then the last recorded one, then we reset the echo trains counter
        if avg_idx > last_avg:
            last_et = 0

        echo_train_mapping.append(
            {
                'acq_idx':          acq_idx,
                'id':               pat_id,
                'fname':            mrd_fpath.name, 
                'avg_idx':          avg_idx,
                'col_idx':          col_idx,
                'slice_idx':        slice_idx,
                'echo_train_idx':   last_et,
                'inner_et_counter': inner_et_counter,
            }
        )
        # LOGGER.info(f"Added: acq_idx: {acq_idx}, pat_id: {pat_id}, avg_idx: {avg_idx}, col_idx: {col_idx}, slice_idx: {slice_idx}, echo_train_idx: {last_et}, inner_et_counter: {inner_et_counter}")
        last_avg = avg_idx      # update the last average index
        last_slice = slice_idx
        inner_et_counter += 1
        # if acq_idx > 3500:
        #     break
    return echo_train_mapping


def find_mrd_fpath(id: str, mrd_root_dir: Path) -> Path:
    t2_tse_files = list(mrd_root_dir.glob('meas_MID*2.mrd'))
    LOGGER.info(f"Found {len(t2_tse_files)} .mrd files in {mrd_root_dir} for patient ID {id}.")
    if len(t2_tse_files) == 0:
        raise FileNotFoundError(f"No .mrd files found in {mrd_root_dir} for patient ID {id}.")
    [LOGGER.info(f) for f in t2_tse_files]
    
    good_files = []     # as in: t2 large files
    for mrd_file in t2_tse_files:
        LOGGER.info(f"Processing {mrd_file}")
        file_size = mrd_file.stat().st_size  # size in bytes
        LOGGER.info(f"File size: {file_size / (1024**2):.2f} MB")
        if file_size < 1 * 1024**3:
            LOGGER.warning(f"Skipping {mrd_file}, size {file_size} bytes is smaller than 1GB.")
            continue
        else: # file size is larger than 1GB
            good_files.append(mrd_file)
            LOGGER.info(f"Adding {mrd_file} to the list of good files.")
        LOGGER.info(f"Processing {mrd_file} (size: {file_size} bytes)")

    # show the user the .mrd options and ask for input in which to select with the index and return that one.
    if len(good_files) == 1:
        LOGGER.info(f"Only one good file found: {good_files[0]}")
        return good_files[0]
    elif len(good_files) > 1:
        LOGGER.warning(f"Multiple good files found: {good_files}")
        LOGGER.info("Please select the file to process by entering the index (0, 1, 2, ...):")
        for i, f in enumerate(good_files):
            LOGGER.info(f"{i}: {f}")
        while True:
            try:
                idx = int(input("Enter the index of the file to process: "))
                if idx < 0 or idx >= len(good_files):
                    raise ValueError("Index out of range.")
                return good_files[idx]
            except ValueError as e:
                LOGGER.warning(f"Invalid input: {e}. Please enter a valid index.")
    else: 
        LOGGER.warning(f"No good files found in {mrd_root_dir} for patient ID {id}.")
        # raise file not found error
        raise FileNotFoundError(f"No good .mrd files found in {mrd_root_dir} for patient ID {id}.")

    return None


def main():
    """
    Main workflow:
      1. Define the list of patients to process.
      2. Create/verify the SQLite database table.
      3. Process each patient by extracting acquisition metadata and inserting it into the DB.
    """
    DEBUG = False

    # All patient IDs to consider for Uncertainty Quantification
    pat_ids = [
        # '0003_ANON5046358',     # DONE    # PART 1
        # '0004_ANON9616598',     # DONE    # PART 1
        # '0005_ANON8290811',     # DONE    # PART 1
        # '0006_ANON2379607',     # DONE    # PART 1
        # '0007_ANON1586301',     # DONE    # PART 1
        # '0008_ANON8890538',     # DONE    # PART 1
        # '0010_ANON7748752',     # DONE    # PART 1
        # '0011_ANON1102778',     # DONE    # PART 1
        # '0012_ANON4982869',     # DONE    # PART 1
        # '0013_ANON7362087',     # DONE    # PART 1
        # '0014_ANON3951049',     # DONE    # PART 1
        # '0015_ANON9844606',     # DONE    # PART 1
        # '0018_ANON9843837',     # DONE    # PART 1
        # '0019_ANON7657657',     # DONE    # PART 1
        # '0020_ANON1562419',     # DONE    # PART 1
        # '0021_ANON4277586',     # DONE    # PART 1
        # '0023_ANON6964611',     # DONE    # PART 1
        # '0024_ANON7992094',     # DONE    # PART 1
        # '0026_ANON3620419',     # DONE    # PART 1
        # '0027_ANON9724912',     # DONE    # PART 1
        # '0028_ANON3394777',           # PART 2
        # '0029_ANON7189994',           # PART 2
        # '0030_ANON3397001',           # PART 2
        # '0031_ANON9141039',           # PART 2
        # '0032_ANON7649583',           # PART 2
        # '0033_ANON9728185',           # PART 2
        # '0035_ANON3474225',           # PART 2
        # '0036_ANON0282755',           # PART 2
        # '0037_ANON0369080',           # PART 2
        # '0039_ANON0604912',           # PART 2
        # '0042_ANON9423619',           # PART 2
        # '0043_ANON7041133',           # PART 2
        # '0044_ANON8232550',           # PART 2
        # '0045_ANON2563804',           # PART 2
        # '0047_ANON3613611',           # PART 2
        # '0048_ANON6365688',           # PART 2
        # '0049_ANON9783006',           # PART 2
        # '0051_ANON1327674',           # PART 2
        # '0052_ANON9710044',           # PART 2
        # '0053_ANON5517301',           # PART 2
        # '0055_ANON3357872',               # PART 3
        # '0056_ANON2124757',               # PART 3
        # '0057_ANON1070291',               # PART 3
        # '0058_ANON9719981',               # PART 3
        # '0059_ANON7955208',               # PART 3
        # '0061_ANON7642254',               # PART 3
        # '0062_ANON0319974',               # PART 3
        # '0063_ANON9972960',               # PART 3
        # '0064_ANON0282398',               # PART 3
        # '0067_ANON0913099',               # PART 3
        # '0068_ANON7978458',               # PART 3
        # '0069_ANON9840567',               # PART 3
        # '0070_ANON5223499',               # PART 3
        # '0071_ANON9806291',               # PART 3
        # '0073_ANON5954143',               # PART 3
        # '0075_ANON5895496',               # PART 3
        # '0076_ANON3983890',               # PART 3
        # '0077_ANON8634437',               # PART 3
        # '0078_ANON6883869',               # PART 3
        # '0079_ANON8828023',               # PART 3
        # '0080_ANON4499321',           # PART 4
        # '0081_ANON9763928',           # PART 4
        # '0082_ANON6073234',           # PART 4
        # '0083_ANON9898497',           # PART 4
        # '0084_ANON6141178',           # PART 4
        # '0085_ANON4535412',           # PART 4
        # '0086_ANON8511628',           # PART 4
        # '0087_ANON9534873',           # PART 4
        # '0088_ANON9892116',           # PART 4
        # '0089_ANON9786899',           # PART 4
        # '0090_ANON0891692',           # PART 4
        # '0092_ANON9941969',           # PART 4
        # '0093_ANON9728761',           # PART 4
        # '0094_ANON8024204',           # PART 4
        # '0095_ANON4189062',           # PART 4
        # '0097_ANON5642073',           # PART 4
        # '0103_ANON8583296',           # PART 4
        # '0104_ANON7748630',           # PART 4
        # '0105_ANON9883201',           # PART 4
        # '0107_ANON4035085',           # PART 4
        '0108_ANON0424679',              # PART 5
        '0109_ANON9816976',              # PART 5
        '0110_ANON8266491',              # PART 5
        '0111_ANON9310466',              # PART 5
        '0112_ANON3210850',              # PART 5
        '0113_ANON9665113',              # PART 5
        '0115_ANON0400743',              # PART 5
        '0116_ANON9223478',              # PART 5
        '0118_ANON7141024',              # PART 5
        '0119_ANON3865800',              # PART 5
        '0120_ANON7275574',              # PART 5
        '0121_ANON9629161',              # PART 5
        '0123_ANON7265874',              # PART 5
        '0124_ANON8610762',              # PART 5
        '0125_ANON0272089',              # PART 5
        '0126_ANON4747182',              # PART 5
        '0127_ANON8023509',              # PART 5
        '0128_ANON8627051',              # PART 5
        '0129_ANON5344332',              # PART 5
        '0135_ANON9879440',              # PART 5
        '0136_ANON8096961',           # PART 6
        '0137_ANON8035619',           # PART 6
        '0138_ANON1747790',           # PART 6
        '0139_ANON2666319',           # PART 6
        '0140_ANON0899488',           # PART 6
        '0141_ANON8018038',           # PART 6
        '0142_ANON7090827',           # PART 6
        '0143_ANON9752849',           # PART 6
        '0144_ANON2255419',           # PART 6
        '0145_ANON0335209',           # PART 6
        '0146_ANON7414571',           # PART 6
        '0148_ANON9604223',           # PART 6
        '0149_ANON4712664',           # PART 6
        '0150_ANON5824292',           # PART 6
        '0152_ANON2411221',           # PART 6
        '0153_ANON5958718',           # PART 6
        '0155_ANON7828652',           # PART 6
        '0157_ANON9873056',           # PART 6
        '0159_ANON9720717',           # PART 6
        '0160_ANON3504149'            # PART 6
    ]
    # pat_ids = [pid.split('_')[0] for pid in patients]

    cfg = {
        'mrd_root_dir': Path(r'\\zkh\appdata\research\Radiology\fastMRI_PCa\03_data\002_processed_pst_ksp_umcg_backup\data\pat_data'),
        'db_fpath': Path('databases/master_habrok_20231106_v2.db'),
        'table_name': 'echo_train_mapping',
        'logdir': Path('logs'),
        'logfile': 'echo_train_mapping.log',
    }

    tablename = cfg['table_name'] + '_debug' if DEBUG else cfg['table_name']
    
    # if the table exists we delete it and create a new one.
    if DEBUG:
        conn = sqlite3.connect(cfg['db_fpath'])
        c = conn.cursor()
        c.execute(f"DROP TABLE IF EXISTS {tablename}")
        conn.commit()
        conn.close()
        LOGGER.warning(f"Table '{tablename}' dropped from database '{cfg['db_fpath']}'.")
    create_db_table_if_not_exists(cfg['db_fpath'], tablename)

    # Process each patient.
    for id in pat_ids:
        # Create the mapping
        echo_train_mapping = process_patient_mrd(id, cfg)
        df = pd.DataFrame(echo_train_mapping)

        # Insert the data into the database
        conn = sqlite3.connect(str(cfg['db_fpath']))
        df.to_sql(tablename, con=conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        LOGGER.info(f"Inserted {len(df)} rows into table '{tablename}'.")


if __name__ == "__main__":

    LOGGER = get_logger(
        name="echo_train_mapping",
        log_dir=Path("logs"),
        level=logging.DEBUG
    )

    main()
