def clean_data(df):
    df.dropna(subset=['Age', 'Embarked'], inplace=True)
    return df