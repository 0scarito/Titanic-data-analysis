import sys, data_loading, data_cleaning, eda, correlation, logistic_regression, NN

def main():

    if len(sys.argv) < 2:
        print("Usage: python main.py 'step' ")
        print("Available steps: load, clean, eda, correlation, logreg (stands for logistic regression), neural_network")
        sys.exit(1)
        
    step = sys.argv[1]
    file_path = 'titanic.csv'

    if step == 'load':
        df = data_loading.load_data(file_path)
        print(df.head())
    elif step == 'clean':
        df = data_loading.load_data(file_path)
        df = data_cleaning.clean_data(df)
        print(df.head())
    elif step == 'eda':
        df = data_loading.load_data(file_path)
        df = data_cleaning.clean_data(df)
        eda.perform_eda(df)
    elif step == 'correlation' :
        df = data_loading.load_data(file_path)
        df = data_cleaning.clean_data(df)
        correlation.correlation_analysis(df)
    elif step == 'logreg' :
        df = data_loading.load_data(file_path)
        df = data_cleaning.clean_data(df)
        logistic_regression.regression(df)        
    elif step == 'neural_network':
        df = data_loading.load_data(file_path)
        df = data_cleaning.clean_data(df)
        NN.neural_network(df)
    else:
        print("Invalid step. Available steps: load, clean, eda, correlation, logreg (stands for logistic regression), neural_network")

if __name__ == '__main__':
    main()