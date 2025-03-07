import sys
import os
import glob
import pandas as pd

def main():
    # Check if the directory argument is provided
    if len(sys.argv) < 2:
        print("Usage: python combine_csv.py [directory] [optional: sort_columns]")
        sys.exit(1)

    # Use the first argument as the target directory
    target_dir = sys.argv[1]

    # Validate that the provided directory exists
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a valid directory.")
        sys.exit(1)

    # Optional sort columns argument
    sort_columns = sys.argv[2:] if len(sys.argv) > 2 else []

    # Construct the pattern for glob
    csv_pattern = os.path.join(target_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    # Check if any CSV files were found
    if not csv_files:
        print(f"No CSV files found in directory: {target_dir}")
        sys.exit(1)

    # Create a list to hold DataFrames
    dfs = []

    # Loop through the list of CSV files and read each one
    for file in csv_files:
        if file.endswith("combined_csv.csv"):
            continue
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort the DataFrame if sort columns are provided
    if sort_columns:
        missing_columns = [col for col in sort_columns if col not in combined_df.columns]
        if missing_columns:
            print(f"Warning: Columns {missing_columns} not found in the DataFrame. Skipping sorting for these columns.")
            sort_columns = [col for col in sort_columns if col in combined_df.columns]
        if sort_columns:
            combined_df = combined_df.sort_values(by=sort_columns)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(f"{target_dir}/combined_csv.csv", index=False)
    print("All CSV files have been combined into 'combined_csv.csv'.")

if __name__ == "__main__":
    main()
