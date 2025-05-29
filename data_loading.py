import pandas as pd

file_name = 'titanic.csv'

def load_data(file_name):
    df = pd.read_csv(file_name)
    df.index = range(1, len(df) + 1)
    return df