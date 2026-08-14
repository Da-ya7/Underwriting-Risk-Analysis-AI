"""
MySQL connection + table setup.

Set creds in .env (copy .env.example -> .env, fill in).
Run once: python -m app.db   (creates the database + table if missing)
"""

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 3306)),
}

DB_NAME = os.getenv("DB_NAME", "underwriting_ai")


def get_connection(with_db=True):
    cfg = dict(DB_CONFIG)
    if with_db:
        cfg["database"] = DB_NAME
    return mysql.connector.connect(**cfg)


def _add_column_if_missing(cur, table, col_def):
    """
    col_def example: 'document_blob LONGBLOB'.

    Safe to re-run. Ignores duplicate column errors.
    """
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except Error as e:
        if e.errno != 1060:  # Duplicate column
            raise


def init_db():
    # Create database if missing
    conn = get_connection(with_db=False)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    conn.commit()
    cur.close()
    conn.close()

    # Create table if missing
    conn = get_connection()
    cur = conn.cursor()

    # Users table — auth (bcrypt hash only, never plain password)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) NOT NULL DEFAULT 'client',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS proposals (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(255),
            insurance_type VARCHAR(50),

            country_code VARCHAR(5) DEFAULT 'IN',
            doc_type VARCHAR(30) DEFAULT 'aadhaar',

            raw_input JSON,
            confidence FLOAT,
            risk_score FLOAT,
            reasoning_summary TEXT,
            risk_factors JSON,
            positive_factors JSON,

            status VARCHAR(30) DEFAULT 'PENDING',

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Migration-safe additions for older databases
    _add_column_if_missing(cur, "proposals", "country_code VARCHAR(5) DEFAULT 'IN'")
    _add_column_if_missing(cur, "proposals", "doc_type VARCHAR(30) DEFAULT 'aadhaar'")

    _add_column_if_missing(cur, "proposals", "document_blob LONGBLOB")
    _add_column_if_missing(cur, "proposals", "document_filename VARCHAR(255)")
    _add_column_if_missing(cur, "proposals", "document_mimetype VARCHAR(100)")
    _add_column_if_missing(cur, "proposals", "extracted_fields JSON")
    _add_column_if_missing(cur, "proposals", "validation_results JSON")
    _add_column_if_missing(cur, "proposals", "user_id INT")

    conn.commit()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            make VARCHAR(100), model VARCHAR(100), year INT,
            vehicle_type VARCHAR(50), engine_cc INT, fuel_type VARCHAR(30),
            vehicle_value FLOAT, safety_features TINYINT, anti_theft TINYINT,
            color VARCHAR(30),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # --- Flat model: one proposal = one vehicle (matches batch_router.py /
    # frontend, which were both built this way). proposals.vehicle_id links
    # each proposal to its one vehicle. ---
    _add_column_if_missing(cur, "proposals", "vehicle_id INT")
    conn.commit()

    # fleet_group_id: NULL for a single-vehicle submission. When N vehicles
    # are submitted together (batch/bulk), every resulting proposal row
    # shares the same fleet_group_id (a UUID generated once per submission
    # call) -- lets the frontend group them into "one proposal" while each
    # vehicle still keeps its own independent AI risk_score/factors row.
    _add_column_if_missing(cur, "proposals", "fleet_group_id VARCHAR(36)")
    conn.commit()

    try:
        cur.execute("CREATE INDEX idx_proposals_fleet_group ON proposals (fleet_group_id)")
        conn.commit()
    except Error as e:
        if e.errno != 1061:
            raise

    # Index speeds up the duplicate-proposal check (user_id + insurance_type + status)
    try:
        cur.execute("CREATE INDEX idx_proposals_user_insurance ON proposals (user_id, insurance_type, status)")
        conn.commit()
    except Error as e:
        if e.errno != 1061:  # Duplicate key name -> index already exists, fine
            raise

    # Index speeds up direct per-user vehicle lookups
    try:
        cur.execute("CREATE INDEX idx_vehicles_user_id ON vehicles (user_id)")
        conn.commit()
    except Error as e:
        if e.errno != 1061:
            raise

    # Index speeds up the proposals -> vehicles join
    try:
        cur.execute("CREATE INDEX idx_proposals_vehicle_id ON proposals (vehicle_id)")
        conn.commit()
    except Error as e:
        if e.errno != 1061:
            raise

    # FK constraint: proposals.vehicle_id -> vehicles.id
    # RESTRICT — a vehicle must not be hard-deleted while a proposal (the
    # audit record) still references it. NULL allowed (life proposals have
    # vehicle_id=NULL), FK only enforces on non-null values.
    try:
        cur.execute("""
            ALTER TABLE proposals
            ADD CONSTRAINT fk_proposals_vehicle_id
            FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
            ON DELETE RESTRICT ON UPDATE RESTRICT
        """)
        conn.commit()
    except Error as e:
        # Dup FK constraint -> already exists, fine.
        # MySQL: errno 1826. MariaDB: errno 1005 wrapping "121" in msg.
        if not (e.errno == 1826 or (e.errno == 1005 and "121" in str(e))):
            raise

    # version_root_id: mentor's "tran_id" concept. Every proposal's FIRST
    # submission gets version_root_id = its own id (set right after insert,
    # in the submit endpoints, since MySQL can't self-reference during the
    # same INSERT). When a client edits and resubmits, the NEW row keeps
    # the SAME version_root_id as the original -- so "give me the latest
    # version of proposal X" = the row with MAX(id) sharing that root.
    # Old edited-over rows are marked status='SUPERSEDED' and hidden from
    # normal list views, but never deleted (audit trail).
    _add_column_if_missing(cur, "proposals", "version_root_id INT")
    conn.commit()

    # Backfill: any row from before this column existed has version_root_id
    # NULL -- treat it as its own root (a proposal that's never been edited).
    # Safe to re-run: only touches rows still NULL.
    cur.execute("UPDATE proposals SET version_root_id = id WHERE version_root_id IS NULL")
    conn.commit()

    try:
        cur.execute("CREATE INDEX idx_proposals_version_root ON proposals (version_root_id)")
        conn.commit()
    except Error as e:
        if e.errno != 1061:
            raise

    cur.close()
    conn.close()


if __name__ == "__main__":
    try:
        init_db()
        print("DB + table ready.")
    except Error as e:
        print("DB setup failed:", e)