import json
import pymysql

def main():
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="task_info",
        port = 44444
    )
    cur = db.cursor()
    cur.execute("""
      SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME
      FROM information_schema.STATISTICS
      WHERE TABLE_SCHEMA = %s
      ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
    """, (db.db.decode('utf-8'),))
    schema = {}
    for tbl, idx, col in cur:
        schema.setdefault(tbl, {}).setdefault(idx, []).append(col)

    out = { tbl: list(idxs.values()) for tbl, idxs in schema.items() }
    with open("idx_schema.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote idx_schema.json")

if __name__ == "__main__":
    main()
