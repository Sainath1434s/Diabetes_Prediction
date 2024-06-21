import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def train_model(X_train, y_train):
    """Train a Random Forest model with imputed missing values."""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('random_forest', RandomForestClassifier())  # Use Random Forest as the classifier
    ])

    pipeline.fit(X_train, y_train.values.ravel())  # Use values.ravel() to reshape y_train
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return y_pred, accuracy, report, matrix

def save_model(model, filepath):
    """Save the trained model to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def plot_confusion_matrix(matrix, class_names):
    """Plot the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test, y_pred_prob):
    """Plot the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("../assets/analyzing.png")
    plt.show()

def main():
    # Load the data
    X_train = load_data('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/X_train.csv')
    X_test = load_data('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/X_test.csv')
    y_train = load_data('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/y_train.csv')
    y_test = load_data('C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/data/processed/y_test.csv')

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred, accuracy, report, matrix = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    print(matrix)

    # Plot the confusion matrix
    class_names = ['Class 0', 'Class 1']  # Adjust based on your class names
    plot_confusion_matrix(matrix, class_names)

    # Plot the ROC curve
    y_pred_prob = model.predict_proba(X_test)
    plot_roc_curve(y_test, y_pred_prob)

    # Save the model
    save_model(model, 'C:/Users/gemba/OneDrive/Desktop/workspace/diabetes-prediction/model/RandomForest_model.joblib')

if __name__ == "__main__":
    main()
