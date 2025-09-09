#!/usr/bin/env python3
import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, user='wuy', database='tpch_sf1')
cursor = conn.cursor()

# Get column names for tables
tables = ['nation', 'supplier', 'lineitem']
for table in tables:
    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' ORDER BY ordinal_position")
    cols = [row[0] for row in cursor.fetchall()]
    print(f"{table}: {cols}")

cursor.close()
conn.close()