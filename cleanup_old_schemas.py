#!/usr/bin/env python3
"""
Clean up old schema-based dataset imports to prepare for database-per-dataset approach
"""

import psycopg2

POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'wuy'
}

def cleanup_old_schemas():
    """Remove old schema-based dataset imports"""
    try:
        # Connect to postgres database
        admin_config = POSTGRESQL_CONFIG.copy()
        admin_config['database'] = 'postgres'
        conn = psycopg2.connect(**admin_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Get list of non-system schemas
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast', 'public')
            ORDER BY schema_name
        """)
        
        schemas = [row[0] for row in cursor.fetchall()]
        
        if not schemas:
            print("✅ No old dataset schemas found to clean up")
            return True
        
        print(f"🧹 Found {len(schemas)} old dataset schemas to clean up:")
        for schema in schemas:
            print(f"   📂 {schema}")
        
        # Ask for confirmation
        response = input(f"\n❓ Remove these {len(schemas)} schemas? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cleanup cancelled")
            return False
        
        # Drop schemas
        for schema in schemas:
            try:
                cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                print(f"   ✅ Removed schema: {schema}")
            except Exception as e:
                print(f"   ✗ Failed to remove schema {schema}: {e}")
        
        print(f"\n🎉 Successfully cleaned up {len(schemas)} old dataset schemas!")
        print("💡 Ready for database-per-dataset import approach")
        return True
        
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
        return False
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    success = cleanup_old_schemas()
    exit(0 if success else 1)