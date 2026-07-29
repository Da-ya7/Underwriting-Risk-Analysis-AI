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
    """col_def e.g. 'document_blob LONGBLOB'. Ignores 'duplicate column' error
    so this is safe to re-run on a DB that already has the column (existing
    installs won't have these new columns until this runs once)."""
    col_name = col_def.split()[0]
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except Error as e:
        if e.errno != 1060:  # 1060 = Duplicate column name
            raise


def init_db():
    conn = get_connection(with_db=False)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    conn.commit()
    cur.close()
    conn.close()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS proposals (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(255),
            insurance_type VARCHAR(50),
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

    # NEW: document storage + doc-validation columns (migration-safe for
    # existing tables created before this change)
    _add_column_if_missing(cur, "proposals", "document_blob LONGBLOB")
    _add_column_if_missing(cur, "proposals", "document_filename VARCHAR(255)")
    _add_column_if_missing(cur, "proposals", "document_mimetype VARCHAR(100)")
    _add_column_if_missing(cur, "proposals", "extracted_fields JSON")
    _add_column_if_missing(cur, "proposals", "validation_results JSON")
    conn.commit()

    cur.close()
    conn.close()


if __name__ == "__main__":
    try:
        init_db()
        print("DB + table ready.")
    except Error as e:
        print("DB setup failed:", e)