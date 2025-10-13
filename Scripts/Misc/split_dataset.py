import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_csv(file_path, split_ratio):
    # Load dataset
    df = pd.read_csv(file_path, sep=";")

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=1 - split_ratio, random_state=42, shuffle=True
    )


    # Format ratio cleanly (e.g., 0.7 -> "0.7")
    ratio_str = f"{split_ratio:.2f}".rstrip("0").rstrip(".")

    # Build output filenames
    train_filename = f"../Datasets/Splits/training_set_{ratio_str}.csv"
    test_filename = f"../Datasets/Splits/testing_set_{ratio_str}.csv"

    # Save CSVs
    train_df.to_csv(train_filename, sep=";", index=False)
    test_df.to_csv(test_filename, sep=";", index=False)

    print(f"✅ Split complete!")
    print(f"Training set: {len(train_df)} rows → {train_filename}")
    print(f"Testing set:  {len(test_df)} rows → {test_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_dataset.py <file.csv> <split_ratio>")
        print("Example: python split_dataset.py dataset.csv 0.8")
        sys.exit(1)

    file_path = sys.argv[1]
    split_ratio = float(sys.argv[2])

    if not (0 < split_ratio < 1):
        print("Error: split_ratio must be between 0 and 1 (e.g., 0.7)")
        sys.exit(1)

    split_csv(file_path, split_ratio)

