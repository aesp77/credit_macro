import sqlite3

conn = sqlite3.connect('data/vol_surfaces.db')
cursor = conn.cursor()

# Check tables
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', [t[0] for t in tables])

# Try vol_surface (singular)
try:
    dates = cursor.execute("SELECT DISTINCT data_date FROM vol_surface ORDER BY data_date DESC LIMIT 5").fetchall()
    print('\nLatest dates in vol_surface:', [d[0] for d in dates])
except Exception as e:
    print(f'\nError with vol_surface: {e}')

conn.close()
