import pyarrow.parquet as pq
import json

table = pq.read_table("databricks_data.parquet")

df = table.to_pandas()
df = df[["instruction", "input", "output"]]

json_string = df.to_json(orient="records")

with open("databricks.json", "w") as f:
    json.dump(json.loads(json_string), f, indent=4)
