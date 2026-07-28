#==============================================
# This component is responsible for ingesting data from a specified path and saving it to the raw data folder. It takes an input path for the raw data file and an output path where the ingested data will be saved. The component ensures that the output directory exists, loads the data, removes duplicates, and saves it in CSV format.
#==============================================

import pandas as pd
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Ingest data from a specified path and save to raw data folder.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output_path', type=str, default='training_pipeline/data/raw/data.csv', help='Path to save the ingested data')
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        data = pd.read_csv(args.input_path)
        print(f"Data loaded from {args.input_path}. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    data = data.drop_duplicates()

    data.to_csv(args.output_path, index=False)
    print(f"Data saved to {args.output_path}")

if __name__ == "__main__":
    main()