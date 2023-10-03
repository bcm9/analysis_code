"""
Script to Generate Value Counts for Each Column in a Dataset
"""

import pandas as pd
from tabulate import tabulate
import sys
import os

def load_data(filepath):
    """Load the dataset from a given filepath."""
    if not os.path.exists(filepath):
        print(f"Error: The specified file {filepath} does not exist.")
        sys.exit()
    return pd.read_csv(filepath)

def generate_value_counts(df):
    """Generate value counts for each column in the dataframe."""
    tables = []
    for column in df.columns:
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'Count']
        table = tabulate(value_counts, headers='keys', tablefmt='pretty')
        tables.append(table)
    return tables

def write_to_file(folder, tables, df):
    """Write the value counts to an output text file."""
    output_filename = "value_counts.txt"
    with open(os.path.join(folder, output_filename), 'w', encoding='utf-8') as f:
        for i, table in enumerate(tables):
            f.write(f"Value Counts for '{df.columns[i]}':\n")
            f.write(table)
            f.write("\n\n")
    print(f"Value counts written to {os.path.join(folder, output_filename)}")

if __name__ == "__main__":
    # Get the folder and filename from the user
    folder = input("Enter the directory path: ")
    filename = input("Enter the filename: ")

    filepath = os.path.join(folder, filename)

    df = load_data(folder+filename)
    tables = generate_value_counts(df)
    write_to_file(folder, tables, df)
