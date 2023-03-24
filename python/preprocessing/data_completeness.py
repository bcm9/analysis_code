# -*- coding: utf-8 -*-
"""
Data completedness script

"""

# Load data
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Select file and folder
filename='C:/Users/bc/test/test_data.csv'

# Load data
clean_df = pd.read_csv(filename)

# Check shape of the dataframe
print(f'The dataframe has {clean_df.shape[0]} rows and {clean_df.shape[1]} columns.')

# Check completeness of each column
completeness = (clean_df.notnull().sum() / len(clean_df)) * 100
print('\nCompleteness of each column:\n')
print(completeness)

# Check data types of each column
print('\nData types of each column:\n')
print(clean_df.dtypes)

# Check summary statistics of each numeric column
summary=clean_df.describe()
print('\nSummary statistics of each numeric column:\n')
print(summary)

# Bar plot completeness
plt.figure(figsize=(10,5))
plt.bar(completeness.index, completeness.values)
plt.xticks(rotation=90)
plt.ylabel('Completeness (%)')
plt.title('Completeness of Data')
plt.show()

# Write completedness and summary dfs to excel file
# Create a pandas Excel writer
timestamp = datetime.datetime.now().strftime("%Y%m%d")
name = "completeness_{}.xlsx".format(timestamp)
writer = pd.ExcelWriter(name, engine='xlsxwriter')

# Write the dfs to excel file
completeness.to_excel(writer, sheet_name='Completeness', index=True)
summary.to_excel(writer, sheet_name='Summary', index=True)

# Save the Excel file
writer.save()
writer.close()
