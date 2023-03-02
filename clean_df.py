# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 20:35:15 2023

Generic script for cleaning data

@author: bcm9
"""
#######################################################################
# 0. LOAD DATA
########################################################################
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Select file and folder
filename='test.csv'
progress_folder='C:/Users/bcm9/test/raw/'

# Load data
unclean_df = pd.read_csv(progress_folder+filename)

#######################################################################
# 1. CHECK MISSING DATA
#######################################################################
timestamp = datetime.datetime.now().strftime("%Y%m%d")
# Count missing values in each column
missing_values = unclean_df.isnull().sum()
missing_values = missing_values.sort_values(ascending=False)

# Export to .csv
name = "missing_values_{}.csv".format(timestamp)
missing_values.to_csv(progress_folder+name,index=True)

# Bar plot missing values
missing_values_filtered = missing_values[missing_values > 0]
sns.barplot(x=missing_values_filtered.index, y=missing_values_filtered.values)
# Plot formatting
plt.xlabel('Columns')
plt.ylabel('Missing Values (#)')
plt.title('Missing Values in Dataset')
plt.xticks(rotation=90, fontsize=7)
# Show plot
plt.show()

# Fill in missing values
clean_df=unclean_df
clean_df.fillna("NaN", inplace=True)

#######################################################################
# 2. CHECK FOR DUPLICATES
#######################################################################
# Check for duplicates
duplicates = clean_df.T.duplicated()

# Count n duplicates, print
n_duplicates = duplicates.sum()
print("N duplicates= ", n_duplicates)

# Get names of duplicate columns
duplicated_cols = clean_df.columns[duplicates].tolist()

# Drop duplicate columns
clean_df.drop(duplicated_cols, axis=1, inplace=True)

#######################################################################
# 3. SUMMARISE DATA, CHECK TYPES
#######################################################################
df_summary=(clean_df.describe())
df_types=(clean_df.dtypes)

#######################################################################
# 4. CHECK FOR MISCODING ERRORS
#######################################################################
# Regex pattern to match miscoding errors
pattern = r'[^\x00-\x7F]+'  # Find non-ASCII characters

# Loop over columns in df, count errors
for col in clean_df.columns:
    # Convert col name to string
    clean_df[col] = clean_df[col].astype(str)
    errors = clean_df[col].str.contains(pattern)
    num_errors = errors.sum()
    print("N miscoding errors in {}: {}".format(col, num_errors))

#######################################################################
# 5. EXPORT CLEAN DATA
#######################################################################
timestamp = datetime.datetime.now().strftime("%Y%m%d")
clean_filename = "clean_data_{}.csv".format(timestamp)
clean_folder='C:/Users/bcm9/test/clean/'
clean_df.to_csv(clean_folder+clean_filename,index=False)