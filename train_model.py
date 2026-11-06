"""
Diabetes Prediction Model Training Script
==========================================
This script trains a Random Forest Classifier on the Pima Indians Diabetes Dataset
and saves the trained model and scaler for use in the web application.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='diabetes.csv'):
    """Load the Pima Indians Diabetes dataset"""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded successfully")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {df.shape[1] - 1}")
    print(f"  - Samples: {df.shape[0]}")
    
    return df

def explore_data(df):
    """Explore and display dataset statistics"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nClass Distribution:")
    print(df['Outcome'].value_counts())
    print(f"  - No Diabetes: {(df['Outcome']==0).sum()} ({(df['Outcome']==0).sum()/len(df)*100:.1f}%)")
    print(f"  - Diabetes: {(df['Outcome']==1).sum()} ({(df['Outcome']==1).sum()/len(df)*100:.1f}%)")
    
    print("\nMissing Values:")
    print(df.isnull().sum())

def prepare_data(df):
    """Prepare data for training"""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data split completed")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Feature scaling completed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_models(X_train, X_test, y_train, y_test):
    """Train and compare different models"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"  ✓ Accuracy: {accuracy*100:.2f}%")
        print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
        print(f"  ✓ CV Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    return results

def save_best_model(results, scaler):
    """Save the best performing model"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Select best model based on ROC-AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"Best Model: {best_model_name}")
    print(f"  - Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
    print(f"  - ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    # Save model and scaler
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print(f"\n✓ Model saved as 'model.pkl'")
    print(f"✓ Scaler saved as 'scaler.pkl'")
    
    return best_model_name, results[best_model_name]

def display_feature_importance(model, feature_names):
    """Display feature importance for Random Forest"""
    if hasattr(model['model'], 'feature_importances_'):
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        importances = model['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeatures ranked by importance:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("DIABETES PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_data('diabetes.csv')
    
    # Explore data
    explore_data(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Save best model
    best_name, best_model = save_best_model(results, scaler)
    
    # Display feature importance
    display_feature_importance(best_model, feature_names)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run 'python app.py' to start the web application")
    print("  2. Open http://localhost:5000 in your browser")
    print("  3. Test the diabetes prediction system")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
