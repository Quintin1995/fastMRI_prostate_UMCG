import sqlite3
import logging
from pathlib import Path
import subprocess


# def transfer_pat_data_to_hpc1(
#     anon_id: str,
#     pat_dir: Path,
#     logger: logging.Logger,
#     conn: sqlite3.Connection,
#     username: str,
#     hostname: str,
#     destination_path: str,
#     **kwargs,
# ) -> None:
#     """
#     Transfers the specified patient's data to the Habrok server using rsync,
#     excluding the 'mrds' directory. Checks the database to ensure all required
#     data conditions are met before initiating the transfer. Updates the database
#     upon successful transfer.

#     Parameters:
#     anon_id (str): Anonymized ID of the patient.
#     pat_dir (Path): Directory of the patient's data.
#     logger (logging.Logger): Logger for error messages.
#     cur (sqlite3.Cursor): Cursor object for executing SQL queries.
#     conn (sqlite3.Connection): Connection object for connecting to the database.
#     cfg (dict): Configuration parameters for the HPC server.

#     Raises:
#     Exception: If an error occurs during data transfer or database operations.
#     """
#     hpc_connect_str = f"{username}@{hostname}:{destination_path}"  
#     exclude_dir = "--exclude=mrds"  # Excludes the 'mrds' directory

#     try:
#         # Check if all required data is present
#         cur.execute("""
#             SELECT has_all_dicoms, has_h5, has_niftis
#             FROM kspace_dset_info
#             WHERE anon_id = ?
#         """, (anon_id,))
#         result = cur.fetchone()
#         if result and all(int(field) == 1 for field in result):
#             # Execute rsync command
#             rsync_command = ["rsync", "-avz", exclude_dir, str(pat_dir), hpc_connect_str]
#             subprocess.run(rsync_command, check=True)

#             # Update database upon successful transfer
#             cur.execute("""
#                 UPDATE kspace_dset_info
#                 SET transfered_to_habrok = 1
#                 WHERE anon_id = ?
#             """, (anon_id,))
#             conn.commit()

#             logger.info(f"Successfully transferred patient data for {anon_id} to Habrok and updated the database.")
#         else:
#             logger.warning(f"Patient {anon_id} does not have all required data for transfer. Missing fields: {result}")

#     except subprocess.CalledProcessError as e:
#         logger.error(f"Failed to transfer patient data for {anon_id}. Error: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"An error occurred while processing patient {anon_id}. Error: {e}")
#         raise
    
    
def transfer_pat_data_to_hpc(
    anon_id: str,
    pat_dir: Path,
    logger: logging.Logger,
    conn: sqlite3.Connection,
    username: str,
    hostname: str,
    destination_path: str,
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
    exclude_dir = "--exclude=mrds"  # Excludes the 'mrds' directory

    try:
        with conn:
            cur = conn.cursor()
            # Check if all required data is present
            cur.execute("""
                SELECT has_all_dicoms, has_h5, has_niftis
                FROM kspace_dset_info
                WHERE anon_id = ?
            """, (anon_id,))
            result = cur.fetchone()
            if result and all((int(field) if field is not None else 0) == 1 for field in result):
                # Execute rsync command
                rsync_command = ["rsync", "-avz", exclude_dir, str(pat_dir), hpc_connect_str]
                subprocess.run(rsync_command, check=True)

                # Update database upon successful transfer
                cur.execute("""
                    UPDATE kspace_dset_info
                    SET transferred_to_habrok = 1
                    WHERE anon_id = ?
                """, (anon_id,))

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
    

# def get_study_date(
#     cur: sqlite3.Cursor,
#     seq_id: str,
#     tablename: str,
# ) -> str:
#     """
#     Retrieves the study date for a patient based on the sequence ID.
    
#     Parameters:
#     cur (sqlite3.Cursor): Database cursor.
#     tablename (str): Name of the table in the database.
#     seq_id (str): The sequence ID of the patient.
    
#     Returns:
#     str: The study date extracted from the database.
#     """
#     query = f"SELECT mri_date FROM {tablename} WHERE seq_id = ?"
#     study_date = cur.execute(query, (seq_id,)).fetchone()[0]
#     return str(study_date)


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


# def update_patient_info_in_db(
#     cur: sqlite3.Cursor,
#     conn: sqlite3.Connection,
#     seq_id: str,
#     anon_id: str,
#     pat_dir: Path,
#     logger: logging.Logger,
#     tablename: str,
#     **kwargs,
# ) -> None:
#     """
#     Update patient information in SQLite database.

#     Parameters:
#     cur: SQLite cursor object.
#     conn: SQLite connection object.
#     seq_id: Sequential ID of the patient.
#     anon_id: Anonymized ID of the patient.
#     pat_dir: Directory path of patient's data.
#     logger: Logger object for logging messages.
#     tablename (str): Name of the table in the database.
#     """
#     # Use parameterized queries to avoid SQL injection
#     cur.execute(f"SELECT COUNT(*) FROM {tablename} WHERE seq_id = ?", (seq_id,))
#     if cur.fetchone()[0] == 0:
#         # Insert a new row if this patient is not in the database
#         cur.execute(f"""
#             INSERT INTO {tablename} (seq_id, anon_id, data_dir)
#             VALUES (?, ?, ?)
#         """, (seq_id, anon_id, str(pat_dir)))
#         logger.info(f"Inserted new patient info in database for patient {seq_id}.")
#     else:
#         # Update existing row for this patient
#         cur.execute(f"""
#             UPDATE {tablename}
#             SET anon_id = ?, data_dir = ?
#             WHERE seq_id = ?
#         """, (anon_id, str(pat_dir), seq_id))
#         logger.info(f"\tUpdated patient info in database for patient {seq_id}.")
#     conn.commit()
    
    
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
