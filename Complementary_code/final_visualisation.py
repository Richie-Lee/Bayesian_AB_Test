import matplotlib.pyplot as plt
import pandas as pd
import os

# Updated base directory
base_dir = "/Users/richie.lee/Desktop/results/real_corrected_effect_sizes/d_0_1"

# Get a list of all Excel files in the base directory
excel_files = [f for f in os.listdir(base_dir) if f.endswith('.xlsx')]

# Read each file into a DataFrame and store them in a list
dfs = []
for file in excel_files:
    file_path = os.path.join(base_dir, file)
    df = pd.read_excel(file_path, index_col=None)

    # Drop any columns that are unnamed (like unnecessary indices)
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    dfs.append(df)

# Merge all DataFrames on the 'sample' column using an outer join
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = merged_df.merge(df, on='sample', how='outer')

# Forward fill to replace null values with the last observed non-null value
merged_df.fillna(method='ffill', inplace=True)

def plot_data(df, x_lim=None):
    # Set 'sample' as the index if it's not already
    if df.index.name != 'sample':
        df = df.set_index('sample')

    # Sort the DataFrame based on 'sample' index
    df.sort_index(inplace=True)

    # Extend data to x_lim if necessary
    if x_lim is not None and df.index.max() < x_lim:
        # For each column, extend the last value to x_lim
        for column in df.columns:
            last_val = df[column].iloc[-1]
            df.loc[x_lim, column] = last_val

    # Sort columns (methods) alphabetically, except for 'sample'
    sorted_columns = sorted(df.columns)
    
    i = 0
    # Plot each method in alphabetical order
    for column in sorted_columns:
        plt.plot(df.index, df[column], label=column)
        i += 1

    # Plot the horizontal line
    plt.axhline(y=0.8, color="black", linestyle="-", linewidth="0.5")
    # plt.axhline(y=0.05, color="grey", linestyle="-", linewidth="0.5")
    # plt.axvline(x = 100, color="black", linestyle="--", linewidth="0.8")

    # Set x-axis limits if x_lim is defined
    if x_lim is not None:
        plt.xlim(0, x_lim)
        
    plt.ylim(0,1)

    # Setting labels and title
    plt.xlabel('Sample')
    plt.ylabel('Power')
    plt.legend()

    # Show the plot
    plt.show()

# Example usage
plot_data(merged_df, x_lim=None)  # Replace 50000 with your desired x_lim value
