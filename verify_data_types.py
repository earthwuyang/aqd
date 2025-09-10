#!/usr/bin/env python3
"""
Verify that imported data has proper types (not all TEXT)
"""

import psycopg2
import sys

def check_database_types(database):
    """Check column types in a database"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='wuy',
            database=database
        )
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📊 Database: {database}")
        print("=" * 60)
        
        total_columns = 0
        text_columns = 0
        numeric_columns = 0
        
        for table in tables:
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table,))
            
            columns = cursor.fetchall()
            print(f"\n  Table: {table}")
            
            for col_name, data_type in columns:
                total_columns += 1
                if data_type.lower() in ['text', 'character varying', 'varchar', 'char']:
                    text_columns += 1
                    marker = "📝"  # Text
                elif data_type.lower() in ['integer', 'bigint', 'smallint', 'numeric', 'real', 'double precision']:
                    numeric_columns += 1
                    marker = "🔢"  # Numeric
                else:
                    marker = "📅"  # Other (date, time, etc.)
                
                print(f"    {marker} {col_name:30s} {data_type}")
        
        cursor.close()
        conn.close()
        
        # Summary
        print(f"\n  Summary:")
        print(f"    Total columns: {total_columns}")
        print(f"    Text columns: {text_columns} ({100*text_columns/total_columns:.1f}%)")
        print(f"    Numeric columns: {numeric_columns} ({100*numeric_columns/total_columns:.1f}%)")
        
        if text_columns == total_columns:
            print("    ❌ ALL COLUMNS ARE TEXT - Data types not properly set!")
            return False
        elif numeric_columns > 0:
            print("    ✅ Data types properly set - mix of text and numeric")
            return True
        else:
            print("    ⚠️ No numeric columns found")
            return False
            
    except psycopg2.OperationalError as e:
        print(f"  ✗ Cannot connect to database '{database}': {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error checking database '{database}': {e}")
        return False

def main():
    """Check all imported databases"""
    databases = ['financial', 'imdb_small', 'Basketball_men', 'northwind', 'sakila', 'world']
    
    print("🔍 Verifying Data Types in Imported Databases")
    
    success_count = 0
    for db in databases:
        if check_database_types(db):
            success_count += 1
    
    print("\n" + "=" * 60)
    if success_count == len(databases):
        print("✅ All databases have proper data types!")
    elif success_count > 0:
        print(f"⚠️ {success_count}/{len(databases)} databases have proper types")
    else:
        print("❌ No databases have proper data types - import may have failed")
    
    return success_count > 0

if __name__ == "__main__":
    sys.exit(0 if main() else 1)