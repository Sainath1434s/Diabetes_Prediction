import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath):
    return pd.read_csv(filepath)

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('random_forest', RandomForestClassifier())  # Use Random Forest as the classifier
    ])

    pipeline.fit(X_train, y_train.values.ravel())  # Use values.ravel() to reshape y_train
    return pipeline

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def main():
    X_train=pd.read_csv('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/X_train.csv')
    X_test= pd.read_csv('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/X_test.csv')
    y_train= pd.read_csv('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/y_train.csv')
    y_test= pd.read_csv('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/y_test.csv')
    model = train_model(X_train, y_train)
    accuracy, report, matrix = evaluate_model(model, X_test, y_test)

    print(f'{accuracy} is the accuracy score')
    print(f'{report} is the classification report')
    print(f'{matrix} is the confusion matrix')

    save_model(model,'C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/model/RandomForest_model.joblib')

if __name__ == "__main__":
    main()
