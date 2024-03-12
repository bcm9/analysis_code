"""
Script to generate value counts for each column in a CSV file.
Input directory and filename.csv, returns text file with value count tables.
"""

import pandas as pd
from tabulate import tabulate
import sys
import os

## LOAD THE DATA FILE
def load_data(filepath):
    # Load the dataset from a given filepath, returns df
    if not os.path.exists(filepath):
        print(f"Error: The specified file {filepath} does not exist.")
        sys.exit()
    else:
        print("Loading and writing data. Please wait")
        
    # Check file extension and load accordingly
    _, file_extension = os.path.splitext(filepath)
    if file_extension == '.csv':
        return pd.read_csv(filepath)
    elif file_extension == '.xlsx':
        return pd.read_excel(filepath)
    else:
        print(f"Error: Unsupported file format {file_extension}")
        sys.exit()
        
## GENERATE VALUE COUNTS IN EACH COLUMN, APPEND TO A LIST OF TABLES
def generate_value_counts(df):
    # Analyses df, generates value counts for each column, and returns list of tables (as strings)
    tables = []
    for column in df.columns:
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'Count']
        table = tabulate(value_counts, headers='keys', tablefmt='pretty')
        tables.append(table)
    return tables

## WRITE VALUE COUNTS TABLES TO OUTPUT TEXT FILE
def write_to_file(folder, tables, df):
    # Write the value counts to an output text file to folder directory
    output_filename = "value_counts.txt"
    with open(os.path.join(folder, output_filename), 'w', encoding='utf-8') as f:
        for i, table in enumerate(tables):
            f.write(f"Value Counts for '{df.columns[i]}':\n")
            f.write(table)
            f.write("\n\n")
    print(f"Value counts written to {os.path.join(folder, output_filename)}")

## CHECK IF SCRIPT IS BEING RUN DIRECTLY
if __name__ == "__main__":
    # Get the folder and filename from the user
    folder = input("Enter the directory path: ")
    filename = input("Enter the filename: ")
    filepath = os.path.join(folder, filename)
    
    # Exectute code block
    df = load_data(folder+filename)
    tables = generate_value_counts(df)
    write_to_file(folder, tables, df)
