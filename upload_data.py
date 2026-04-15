import pandas as pd
from pymongo import MongoClient
import os

MONGODB_URL = os.environ.get("MONGODB_URL")
if not MONGODB_URL:
    raise Exception("MONGODB_URL environment variable not set!")

print("Connecting to MongoDB...")
client = MongoClient(MONGODB_URL)

print("Reading CSV file...")
df = pd.read_csv("notebook/waterQuality1.csv")
print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Remove rows with invalid is_safe values like '#NUM!'
before = len(df)
df = df[pd.to_numeric(df["is_safe"], errors="coerce").notna()].copy()
df["is_safe"] = df["is_safe"].astype(int)
print(f"Removed {before - len(df)} invalid rows. Clean shape: {df.shape}")
print(f"Target distribution:\n{df['is_safe'].value_counts()}")

# Upload to MongoDB
db = client["water_quality"]
collection = db["water_data"]
collection.drop()
collection.insert_many(df.to_dict("records"))

count = collection.count_documents({})
print(f"\nSuccessfully uploaded {count} records to MongoDB!")
print(f"Sample keys: {list(collection.find_one().keys())}")
client.close()
