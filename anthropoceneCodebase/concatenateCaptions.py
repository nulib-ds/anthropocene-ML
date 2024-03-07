import pandas as pd

def concatenate_on_column(dataframe, column_index):
    """
    Concatenates the values of a column in a dataframe into a single string
    """
    return dataframe.iloc[:, column_index].str.cat(sep=' ')

def main():
    # Load the data
    df = pd.read_csv('/home/ysc4337/aerith/anthropocene-reconcile/anthropocene-data/results/image_captioning_extending_output_length.csv')
    print(df)
    # Concatenate the captions using the second column (index 1)
    concatenated_captions = concatenate_on_column(df, 1)

    # Save the concatenated captions
    with open('/home/ysc4337/aerith/anthropocene-reconcile/anthropocene-data/results/concatenated_captions.txt', 'a') as f:
        f.write(concatenated_captions)

main()
