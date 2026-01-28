import pandas as pd
import json

# Load the vehicles JSONL
df = pd.read_json("runs/volvo_s60/vehicles.jsonl", lines=True)

print(f"Total rows: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}\n")

# Check how many rows have the required fields
print("Rows with non-null values:")
print(f" - make: {df['make'].notna().sum()}")
print(f" - model: {df['model'].notna().sum()}")
print(f" - year: {df['year'].notna().sum()}")
print(f" - mileage_km: {df['mileage_km'].notna().sum()}")
print(f" - price_eur: {df['price_eur'].notna().sum()}")
print(f" - fuel: {df['fuel'].notna().sum()}")

# Show a sample row
print("\n" + "="*80)
print("Sample row (first row):")
print("="*80)
row = df.iloc[0].to_dict()
for k, v in row.items():
    if v is not None and v != "" and v != []:
        print(f"{k}: {v}")
