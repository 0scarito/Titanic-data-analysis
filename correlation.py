import matplotlib.pyplot as plt
import seaborn as sns

def correlation_analysis(df):
    # Convert categorical variables into numerical values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})             # 0 for male and 1 for female 
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})      

    # Handle missing values by filling them with the median value to not affect results.
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].median(), inplace=True)

    # Drop columns that are not useful for correlation
    df_corr = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Calculate the correlation matrix
    corr_matrix = df_corr.corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    # Analyze the correlation
    print(corr_matrix['Survived'].sort_values(ascending=False))