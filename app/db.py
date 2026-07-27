"""
MySQL connection + table setup.

Edit DB_CONFIG below with your local MySQL creds before running.
Run once: python -m app.db   (creates the database + table if missing)
"""

import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "iambatman@12",   # <-- CHANGE THIS
    "port": 3306,
}   

DB_NAME = "underwriting_ai"


def get_connection(with_db=True):
    cfg = dict(DB_CONFIG)
    if with_db:
        cfg["database"] = DB_NAME
    return mysql.connector.connect(**cfg)


def init_db():
    # create database if missing
    conn = get_connection(with_db=False)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    conn.commit()
    cur.close()
    conn.close()

    # create table if missing
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
    cur.close()
    conn.close()


if __name__ == "__main__":
    try:
        init_db()
        print("DB + table ready.")
    except Error as e:
        print("DB setup failed:", e)

            