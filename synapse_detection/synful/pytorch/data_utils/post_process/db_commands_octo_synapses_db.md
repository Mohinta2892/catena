## Open DB
```
sqlite3  /mnt/graid/synapse_detection/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/synapse_predictions.db
```

## Schema sqlite queries
#### List all tables
```
.tables
```

#### View the full schema SQL used to create the tables
```
SELECT sql FROM sqlite_schema WHERE type='table';
```

#### Alternatively, showing detailed info for a specific table
```
PRAGMA table_info(pre_sites);
```

### Examples
```bash (syn_save) smohinta-local@cardona-gpu2:/mnt/graid/synapse_detection$ sqlite3  /mnt/graid/synapse_detection/octo_cns_setup_03_octo_cube_all3_same_preid_256_300000/synapse_predictions.db
SQLite version 3.45.1 2024-01-30 16:01:20
Enter ".help" for usage hints.
sqlite> .tables
post_sites        pre_post_mapping  pre_sites 
```

```bash
sqlite> SELECT sql FROM sqlite_schema WHERE type='table';
CREATE TABLE pre_sites (id INTEGER PRIMARY KEY, z REAL, y REAL, x REAL, score REAL)
CREATE TABLE post_sites (id INTEGER PRIMARY KEY, z REAL, y REAL, x REAL)
CREATE TABLE pre_post_mapping (pre_id INTEGER, post_id INTEGER, FOREIGN KEY(pre_id) REFERENCES pre_sites(id), FOREIGN KEY(post_id) REFERENCES post_sites(id))
```

```bash
sqlite> PRAGMA table_info(pre_sites);
0|id|INTEGER|0||1
1|z|REAL|0||0
2|y|REAL|0||0
3|x|REAL|0||0
4|score|REAL|0||0
```

## Basic data preview

### `pre_sites` table
```
SELECT * FROM pre_sites LIMIT 5;
```

### `post_sites` table
```
SELECT * FROM post_sites LIMIT 5;
```

### `pre_post_mapping table
```
SELECT * FROM pre_post_mapping LIMIT 5;
```

## This joins all three tables so you can see the Pre-Coordinate, Post-Coordinate, and Score in a single row.
```
SELECT 
    pre.id AS pre_id,
    pre.score,
    pre.z AS pre_z, pre.y AS pre_y, pre.x AS pre_x,
    post.id AS post_id,
    post.z AS post_z, post.y AS post_y, post.x AS post_x
FROM pre_sites pre
JOIN pre_post_mapping map ON pre.id = map.pre_id
JOIN post_sites post ON post.id = map.post_id
LIMIT 10;
```

### Count synapses
```
SELECT count(*) AS total_synapses FROM pre_sites;
```

### Filter by score
```
SELECT count(*) AS high_conf_synapses 
FROM pre_sites 
WHERE score > 100;
```

### Find synapses in crop
```
SELECT * FROM pre_sites 
WHERE 
    z BETWEEN 4000 AND 5000 
    AND y BETWEEN 10000 AND 12000
    AND x BETWEEN 12000 AND 14000
LIMIT 20;
```

### Python
```
import sqlite3
import pandas as pd

db_path = "/path/to/your/synapse_predictions.db"
conn = sqlite3.connect(db_path)

# Query full pairs
query = """
SELECT 
    pre.id AS pre_id,
    pre.score,
    pre.z AS pre_z, pre.y AS pre_y, pre.x AS pre_x,
    post.z AS post_z, post.y AS post_y, post.x AS post_x
FROM pre_sites pre
JOIN pre_post_mapping map ON pre.id = map.pre_id
JOIN post_sites post ON post.id = map.post_id
LIMIT 100
"""

df = pd.read_sql_query(query, conn)
print(df.head())
conn.close()
```



