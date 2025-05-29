import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):

    # Overall survival rate 
    total_survival = df['Survived'].sum()
    rate_survival = total_survival / len(df) * 100
    plt.figure(figsize=(8, 8))
    plt.pie([total_survival, len(df) - total_survival], labels=['Survived', 'Did not survive'],
        autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FF5733'])
    plt.title(f'Survival Rate: {rate_survival:.2f}%')
    plt.show()

    # Survival rate by feature 
    def visualize_survival_rate(df, column, title):
        survived = df[df['Survived'] == 1][column].value_counts()
        not_survived = df[df['Survived'] == 0][column].value_counts()
        survival_rate = (survived / (survived + not_survived) * 100).sort_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=survival_rate.index, y=survival_rate.values, palette='viridis')
        plt.title(f'Survival Rate by {title}')
        plt.ylabel('Survival Rate (%)')
        plt.xlabel(title)
        plt.show()

    # Survival rate by sex 
    visualize_survival_rate(df, 'Sex', 'Gender')

    # Survival rate by passenger class 
    visualize_survival_rate(df, 'Pclass', 'Passenger Class')

    # Survival rate by embarkation point 
    visualize_survival_rate(df, 'Embarked', 'Embarkation Point')


    # Plot Distribution of survival by age 
    cont_features = ['Age']
    surv = df['Survived'] == 1
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    feature = cont_features[0]
    # Distribution of survival in feature
    sns.histplot(df[~surv][feature], label='Not Survived', color='#e74c3c', ax=ax, kde=True)
    sns.histplot(df[surv][feature], label='Survived', color='#2ecc71', ax=ax, kde=True)
    # Set labels and title
    ax.set_xlabel(feature, fontsize=15)
    ax.set_ylabel('Density', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper right', prop={'size': 12})
    ax.set_title('Distribution of Survival in {}'.format(feature), size=15)
    plt.show()

    # Age distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.show()

