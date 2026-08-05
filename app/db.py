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

    # Index speeds up the duplicate-proposal check (user_id + insurance_type + status)
    try:
        cur.execute("CREATE INDEX idx_proposals_user_insurance ON proposals (user_id, insurance_type, status)")
        conn.commit()
    except Error as e:
        if e.errno != 1061:  # Duplicate key name -> index already exists, fine
            raise

    cur.close()
    conn.close()


if __name__ == "__main__":
    try:
        init_db()
        print("DB + table ready.")
    except Error as e:
        print("DB setup failed:", e)