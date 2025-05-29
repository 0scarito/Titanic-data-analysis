from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

def neural_network(df):
    # Handle missing values with more advanced imputation if necessary
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Convert categorical variables into numerical values
    data = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # Drop columns that are not useful for the model
    data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    # Define features and target
    X = data.drop(columns=['Survived'])
    y = data['Survived']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()  # Standardize the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the neural network architecture with more hyperparameters for tuning
    model = MLPClassifier(max_iter=50, random_state=42)

    # Hyperparameter tuning using Grid Search
    parameter_space = {
        'hidden_layer_sizes': [(32, 16), (64, 32), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    # Best parameters and estimator
    print('Best parameters found:\n', clf.best_params_)
    model = clf.best_estimator_

    # Train the neural network
    history = model.fit(X_train, y_train)

    # Evaluate the neural network
    test_accuracy = model.score(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Confusion matrix and classification report
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    # Plot training loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(history.loss_curve_, label='Train Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Train Accuracy')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Size')
    plt.legend(loc='upper left')
    plt.show()

    # Cross-Validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

