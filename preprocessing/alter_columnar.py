import mysql.connector

# ✅ PolarDB 连接参数（请根据实际修改）
host = "127.0.0.1"
port = 44444
user = "root"
password = ""

# ✅ 你要操作的数据库列表
# database_list = ['airline', 'carcinogenesis', 'credit',
#                  'employee', 'financial', 'geneea',
#                  'hepatitis', 'walmart']

database_list = ['tpch_sf10']

for db in database_list:
    print(f"\n==== Processing database: {db} ====")

    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db
        )
        cursor = conn.cursor()

        # 获取所有表名
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(tables)} tables in {db}")

        # 遍历并修改表注释
        for table in tables:
            try:
                sql = f"ALTER TABLE `{table}` COMMENT='COLUMNAR=1'"
                print(f"  Executing: {sql}")
                cursor.execute(sql)
            except Exception as e:
                print(f"  [ERROR] Failed to alter table {table}: {e}")

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Done with database: {db}")

    except Exception as e:
        print(f"[ERROR] Could not connect to database {db}: {e}")
