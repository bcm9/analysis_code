# long2widedf converts a long format dataframe to a wide format
# argument is df. user enters ID column name of which to group by into command line.
import pandas as pd

def long2widedf(df):
    print('Enter ID column name to group by:')
    id_string=input()
    df=df.groupby(id_string).agg(lambda x: x.tolist())

    # splits columns iteratively into a dict
    df_dict={}
    for column in df.columns:
        df_dict[column]=(pd.DataFrame(df[column].values.tolist(), index=df.index))
        
    # loop through each df in dict, renaming columns (key name + digit), then updating df in the dictionary with the new version   
    for key, df in df_dict.items():
        df.columns = [f"{key}_{i+1}" for i in range(df.shape[1])]
        df_dict[key] = df
        
    # append dataframes in dict to list then concat to dataframe
    df_list = []
    # loop through the dictionary to extract each dataframe
    for key, value in df_dict.items():
        df_list.append(value)
    # concatenate all dataframes in the list along axis 1 (columns)
    wide_df = pd.concat(df_list, axis=1)
    
    return wide_df