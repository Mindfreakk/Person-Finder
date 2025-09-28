import sqlite3, os, json
DB = r"E:\Projects\PersonFinder\personfinder.db"
if not os.path.exists(DB):
    print("DB not found:", DB); raise SystemExit(1)
con = sqlite3.connect(DB)
cur = con.cursor()
print("Tables:")
for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"):
    print(" ", row[0])
print("\nSchemas:")
for tbl in ['people','users','user','search_logs','push_subscriptions']:
    try:
        cur.execute(f"PRAGMA table_info('{tbl}');")
        cols = cur.fetchall()
        if cols:
            print(f"\n{tbl} columns:")
            for c in cols:
                print(" ", c)
    except Exception as e:
        pass
print("\nCounts:")
for tbl in ['people','users','search_logs','push_subscriptions']:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        print(f" {tbl}: {cur.fetchone()[0]}")
    except Exception as e:
        print(f" {tbl}: (no table)")
con.close()
