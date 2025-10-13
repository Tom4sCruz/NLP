import pandas as pd
import argparse
import os

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Remove the 'chef_id' column from a CSV file")
parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
args = parser.parse_args()

input_path = args.input_csv

# ----------------------------
# Load CSV and remove chef_id
# ----------------------------
df = pd.read_csv(input_path, sep=";")  # adjust sep if needed
if "chef_id" in df.columns:
    df = df.drop(columns=["chef_id"])
else:
    print("⚠️  Column 'chef_id' not found in the CSV.")

# ----------------------------
# Save the new CSV
# ----------------------------
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_path = f"../Datasets/Splits/{base_name}_no_chef_id.csv"

df.to_csv(output_path, index=False, sep=";")  # keep same separator
print(f"✅ Saved CSV without 'chef_id' as {output_path}")

