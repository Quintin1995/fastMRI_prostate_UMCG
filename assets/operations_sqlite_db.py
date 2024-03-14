import sqlite3
import logging
from pathlib import Path


def check_mri_date_exists(cur: sqlite3.Cursor, seq_id: str, tablename: str = 'kspace_dset_info') -> bool:
    """
    Check if the MRI date for a specific patient scan is already set in the database.

    Parameters:
    cur (sqlite3.Cursor): Database cursor.
    seq_id (str): The sequence ID of the patient.

    Returns:
    bool: True if the MRI date exists and is not null, False otherwise.
    """
    cur.execute(f"SELECT mri_date FROM {tablename} WHERE seq_id = ?", (seq_id,))
    result = cur.fetchone()
    return result is not None and result[0] is not None


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
    cur: sqlite3.Cursor,
    conn: sqlite3.Connection,
    table: str,
    seq_id: str,
    anon_id: str,
    pat_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Update patient information in SQLite database.

    Parameters:
    cur: SQLite cursor object.
    conn: SQLite connection object.
    table (str): Name of the table in the database.
    seq_id: Sequential ID of the patient.
    anon_id: Anonymized ID of the patient.
    pat_dir: Directory path of patient's data.
    logger: Logger object for logging messages.
    """
    # Use parameterized queries to avoid SQL injection
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE seq_id = ?", (seq_id,))
    if cur.fetchone()[0] == 0:
        # Insert a new row if this patient is not in the database
        cur.execute(f"""
            INSERT INTO {table} (seq_id, anon_id, data_dir)
            VALUES (?, ?, ?)
        """, (seq_id, anon_id, str(pat_dir)))
        logger.info(f"Inserted new patient info in database for patient {seq_id}.")
    else:
        # Update existing row for this patient
        cur.execute(f"""
            UPDATE {table}
            SET anon_id = ?, data_dir = ?
            WHERE seq_id = ?
        """, (anon_id, str(pat_dir), seq_id))
        logger.info(f"\tUpdated patient info in database for patient {seq_id}.")
    conn.commit()
    