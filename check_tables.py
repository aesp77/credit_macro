import sqlite3

conn = sqlite3.connect('data/raw/cds_indices_raw.db')
cursor = conn.cursor()

# Get all tables
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables in cds_indices_raw.db:")
for table in tables:
    print(f"  - {table[0]}")
    # Get column info for each table
    columns = cursor.execute(f"PRAGMA table_info({table[0]})").fetchall()
    print(f"    Columns: {[col[1] for col in columns]}")

conn.close()
