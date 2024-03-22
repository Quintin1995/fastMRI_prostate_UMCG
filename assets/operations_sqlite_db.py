import sqlite3
import logging
from pathlib import Path
import subprocess

    
    
def transfer_pat_data_to_hpc(
    anon_id: str,
    pat_dir: Path,
    logger: logging.Logger,
    conn: sqlite3.Connection,
    username: str,
    hostname: str,
    destination_path: str,
    tablename_ksp: str,
    **kwargs,
) -> None:
    """
    Transfers the specified patient's data to the Habrok server using rsync,
    excluding the 'mrds' directory. Checks the database to ensure all required
    data conditions are met before initiating the transfer. Updates the database
    upon successful transfer.

    Parameters:
    anon_id (str): Anonymized ID of the patient.
    pat_dir (Path): Directory of the patient's data.
    logger (logging.Logger): Logger for error messages.
    conn (sqlite3.Connection): Connection object for connecting to the database.
    cfg (dict): Configuration parameters for the HPC server.

    Raises:
    Exception: If an error occurs during data transfer or database operations.
    """
    hpc_connect_str = f"{username}@{hostname}:{destination_path}"  
    exclude_dir = "--exclude=mrds"  # Excludes the 'mrds' directory from transfer

    try:
        with conn:
            cur = conn.cursor()
            query = f"""
                SELECT has_all_dicoms, has_h5, has_niftis
                FROM {tablename_ksp}
                WHERE anon_id = ?
            """
            cur.execute(query, (anon_id,))
            result = cur.fetchone()
            
            # Check if all required data is present
            if result and all((int(field) if field is not None else 0) == 1 for field in result):
                
                # Sync patient data to Habrok server using rsync
                rsync_command = ["rsync", "-avz", exclude_dir, str(pat_dir), hpc_connect_str]
                subprocess.run(rsync_command, check=True)

                # Update database upon successful transfer
                query = f"""
                    UPDATE {tablename_ksp}
                    SET transfered_to_habrok = 1
                    WHERE anon_id = ?
                """
                cur.execute(query, (anon_id,))
                logger.info(f"Successfully transferred patient data for {anon_id} to Habrok and updated the database.")
            else:
                logger.warning(f"Patient {anon_id} does not have all required data for transfer. Missing fields: {result}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to transfer patient data for {anon_id}. Error: {e}")
        raise
    except sqlite3.Error as e:
        logger.error(f"Database error during operations for patient {anon_id}. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while processing patient {anon_id}. Error: {e}")
        raise


def get_study_date(
    conn: sqlite3.Connection,
    seq_id: str,
    tablename: str
) -> str:
    """
    Retrieves the study date for a patient based on the sequence ID from a specified table.
    
    Parameters:
    conn (sqlite3.Connection): SQLite database connection object.
    seq_id (str): The sequence ID of the patient.
    tablename (str): Name of the table in the database.
    
    Returns:
    str: The study date extracted from the database.
    
    Raises:
    ValueError: If the study date is not found for the given sequence ID.
    """
    query = f"SELECT mri_date FROM {tablename} WHERE seq_id = ?"
    try:
        with conn:
            cur = conn.cursor()
            cur.execute(query, (seq_id,))
            result = cur.fetchone()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Database error: {e}")
    
    if not result or result[0] is None:
        raise ValueError(f"Study date not found for sequence ID {seq_id}")
    
    return str(result[0])


def check_mri_date_exists(conn: sqlite3.Connection, seq_id: str, tablename: str = None) -> bool:
    """
    Check if the MRI date for a specific patient scan is already set in the database.

    Parameters:
    conn (sqlite3.Connection): Database connection.
    seq_id (str): The sequence ID of the patient.

    Returns:
    bool: True if the MRI date exists and is not null, False otherwise.
    """
    assert tablename is not None, "Table name must be provided."
    
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT mri_date FROM {tablename} WHERE seq_id = ?", (seq_id,))
        result = cur.fetchone()
        return result is not None and result[0] is not None
    except sqlite3.Error as e:
        print(f"Databse error occured: {e}")


def connect_to_database(database_path: str):
    """
    Connect to an SQLite database.
    
    Parameters:
    - database_path (str): Path to the SQLite database file.
    
    Returns:
    - conn (sqlite3.Connection): SQLite database connection object.
    """
    conn = sqlite3.connect(database_path)
    return conn
    
    
def update_patient_info_in_db(
    conn: sqlite3.Connection,  # Use the connection object directly
    seq_id: str,
    anon_id: str,
    pat_dir: Path,
    logger: logging.Logger,
    tablename: str,
    **kwargs,
) -> None:
    """
    Update patient information in SQLite database.

    Parameters:
    conn: SQLite connection object.
    seq_id: Sequential ID of the patient.
    anon_id: Anonymized ID of the patient.
    pat_dir: Directory path of patient's data.
    logger: Logger object for logging messages.
    tablename (str): Name of the table in the database.
    """
    try:
        cur = conn.cursor()  # Create a cursor object using the connection
        # Check if this patient is already in the database
        cur.execute(f"SELECT COUNT(*) FROM {tablename} WHERE seq_id = ?", (seq_id,))
        if cur.fetchone()[0] == 0:
            # Insert a new row if this patient is not in the database
            cur.execute(f"""
                INSERT INTO {tablename} (seq_id, anon_id, data_dir)
                VALUES (?, ?, ?)
            """, (seq_id, anon_id, str(pat_dir)))
            logger.info(f"Inserted new patient info in database for patient {seq_id}.")
        else:
            # Update existing row for this patient
            cur.execute(f"""
                UPDATE {tablename}
                SET anon_id = ?, data_dir = ?
                WHERE seq_id = ?
            """, (anon_id, str(pat_dir), seq_id))
            logger.info(f"\tUpdated patient info in database for patient {seq_id}.")

        conn.commit()  # Commit the transaction manually if no exceptions occurred
    except sqlite3.Error as e:
        conn.rollback()  # Rollback the transaction in case of error
        logger.error(f"Failed to update database for patient {seq_id}: {e}")
