import pandas as pd
import os

source_dir = os.path.join(os.environ["HF_HOME"], "hub","datasets--HuggingFaceH4--no_robots", "snapshots",
    "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b","data")
target_dir = os.path.join(".",".keys","datasets--HuggingFaceH4--no_robots")

print(f"source_dir={source_dir}, {target_dir=}")
os.makedirs(target_dir, exist_ok=True)

for fname in os.listdir(source_dir):
    if fname.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(source_dir, fname))
        base = fname.replace(".parquet", "")
        df.to_csv(os.path.join(target_dir, f"{base}.csv"), index=False)
        # 或保存为 jsonl
        # df.to_json(os.path.join(target_dir, f"{base}.jsonl"), orient="records", lines=True)