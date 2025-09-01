import subprocess

# Configurations
remote_host = "relational.fel.cvut.cz"
remote_user = "guest"
remote_password = "ctu-relational"

local_host = "127.0.0.1"
local_port = 44444
local_user = "root"
local_password = ""  # Fill if you set one

# List of databases you want to copy
database_list = ['airline', 'carcinogenesis', 'credit',
                'employee', 'financial', 'geneea', 'hepatitis',
                'walmart']

mysql_database_list = ['Airline', 'Carcinogenesis', 'Credit',
                       'employee', 'financial', 'geneea', 'Hepatitis_std', 
                      'Walmart']

# Output dir for intermediate dumps (optional)
dump_dir = "/home/wuy/datasets/zero-shot-data"

import os
os.makedirs(dump_dir, exist_ok=True)

for i in range(len(mysql_database_list)):
    mysql_db_name = mysql_database_list[i]
    db = database_list[i]
    print(f"== Dumping remote database: {db} ==")
    dump_file = f"{dump_dir}/{db}.sql"

    # 1. Dump from remote
    dump_cmd = [
        "mysqldump",
        f"-h{remote_host}",
        f"-u{remote_user}",
        f"-p{remote_password}",
        "--single-transaction",
        "--skip-lock-tables",
        "--databases", mysql_db_name
    ]

    with open(dump_file, "w") as f:
        ret = subprocess.run(dump_cmd, stdout=f)
        if ret.returncode != 0:
            print(f"[ERROR] Failed to dump {db}")
            continue

    print(f"== Restoring to local PolarDB: {db} ==")

    # 2. Import into local PolarDB
    restore_cmd = [
        "mysql",
        f"-h{local_host}",
        f"-P{local_port}",
        f"-u{local_user}",
    ]
    if local_password:
        restore_cmd.append(f"-p{local_password}")
    
    with open(dump_file, "r") as f:
        ret = subprocess.run(restore_cmd, stdin=f)
        if ret.returncode != 0:
            print(f"[ERROR] Failed to restore {db} to local PolarDB")
        else:
            print(f"[OK] Successfully copied {db}")
