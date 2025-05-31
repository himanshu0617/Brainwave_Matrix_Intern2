import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please download 'creditcard.csv' from Kaggle and place it in the data folder.")
    return pd.read_csv(path)

def preprocess_data(df):
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:,1]
    results['lr'] = {
        'model': lr,
        'report': classification_report(y_test, y_pred_lr, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba_lr),
        'confusion': confusion_matrix(y_test, y_pred_lr),
        'fpr_tpr': roc_curve(y_test, y_proba_lr),
        'y_pred': y_pred_lr
    }
    # Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]
    results['rf'] = {
        'model': rf,
        'report': classification_report(y_test, y_pred_rf, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba_rf),
        'confusion': confusion_matrix(y_test, y_pred_rf),
        'fpr_tpr': roc_curve(y_test, y_proba_rf),
        'y_pred': y_pred_rf
    }
    return results

def plot_results(results):
    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(results['lr']['confusion'], annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Logistic Regression Confusion Matrix')
    sns.heatmap(results['rf']['confusion'], annot=True, fmt='d', ax=axes[1], cmap='Greens')
    axes[1].set_title('Random Forest Confusion Matrix')
    plt.show()
    # ROC Curves
    fpr_lr, tpr_lr, _ = results['lr']['fpr_tpr']
    fpr_rf, tpr_rf, _ = results['rf']['fpr_tpr']
    plt.figure(figsize=(8,6))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def main():
    data_path = os.path.join('data', 'creditcard.csv')
    print('Loading data...')
    df = load_data(data_path)
    # For faster development, use a 5% sample of the data. Remove or adjust for final results.
    df = df.sample(frac=0.05, random_state=42)
    print('Sampled data shape:', df.shape)
    print('Preprocessing data...')
    X, y = preprocess_data(df)
    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print('Balancing data with SMOTE...')
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print('Resampled class distribution:', np.bincount(y_train_res))
    print('Training Logistic Regression...')
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:,1]
    print('Logistic Regression Results:')
    print(classification_report(y_test, y_pred_lr))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba_lr))
    print('Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train_res, y_train_res)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]
    print('Random Forest Results:')
    print(classification_report(y_test, y_pred_rf))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba_rf))
    print('Model training complete.')
    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Logistic Regression Confusion Matrix')
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=axes[1], cmap='Greens')
    axes[1].set_title('Random Forest Confusion Matrix')
    plt.show()
    # ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    print('Done.')

if __name__ == '__main__':
    main()
