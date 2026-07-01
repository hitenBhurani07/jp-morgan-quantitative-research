import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

# Suppress sklearn warnings for clean output
warnings.filterwarnings('ignore')

# Global variables to store the best model and scaler
best_model = None
scaler = None

def train_and_compare_models(csv_path='loan_data.csv'):
    """
    Trains multiple models, compares them, and selects the best one for probability estimation.
    """
    global best_model, scaler
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Ensure the dataset is saved in the same directory.")
        return
        
    # Define features (X) and target (y)
    features = [
        'credit_lines_outstanding', 
        'loan_amt_outstanding', 
        'total_debt_outstanding', 
        'income', 
        'years_employed', 
        'fico_score'
    ]
    X = df[features]
    y = df['default']
    
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("--- Model Comparison ---")
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)
    print(f"Logistic Regression AUC: {lr_auc:.4f}")
    
    # 2. Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train_scaled, y_train) # Using scaled for consistency in this script
    dt_probs = dt_model.predict_proba(X_test_scaled)[:, 1]
    dt_auc = roc_auc_score(y_test, dt_probs)
    print(f"Decision Tree AUC: {dt_auc:.4f}")
    
    # Select the model with the higher AUC for our expected loss function
    if lr_auc >= dt_auc:
        best_model = lr_model
        print("\nSelected Model: Logistic Regression")
    else:
        best_model = dt_model
        print("\nSelected Model: Decision Tree")
        
    # Retrain the best model on the FULL dataset for maximum data utilization
    X_full_scaled = scaler.fit_transform(X)
    best_model.fit(X_full_scaled, y)
    print("Final model trained on all available data.\n")


def calculate_expected_loss(loan_properties: dict) -> float:
    """
    Takes in properties of a loan and outputs the expected loss.
    
    Expected Loss = Probability of Default (PD) * Exposure at Default (EAD) * Loss Given Default (LGD)
    - PD is estimated by the machine learning model.
    - EAD is the loan_amt_outstanding.
    - LGD is (1 - recovery_rate). The prompt states recovery rate is 10%, so LGD is 90% (0.90).
    
    Args:
        loan_properties (dict): Dictionary with keys matching the feature columns.
        
    Returns:
        float: Expected loss in dollars, rounded to 2 decimal places.
    """
    global best_model, scaler
    
    if best_model is None or scaler is None:
        raise ValueError("Model is not trained. Please call train_and_compare_models() first.")
        
    features = [
        'credit_lines_outstanding', 
        'loan_amt_outstanding', 
        'total_debt_outstanding', 
        'income', 
        'years_employed', 
        'fico_score'
    ]
    
    # Extract features in the correct order
    try:
        x_input = [[loan_properties[f] for f in features]]
    except KeyError as e:
        raise ValueError(f"Missing required loan property in dictionary: {e}")
        
    # Scale the input
    x_scaled = scaler.transform(x_input)
    
    # Predict Probability of Default (PD)
    probability_of_default = best_model.predict_proba(x_scaled)[0][1]
    
    # Calculate Expected Loss
    exposure_at_default = loan_properties['loan_amt_outstanding']
    recovery_rate = 0.10
    loss_given_default = 1.0 - recovery_rate
    
    expected_loss = probability_of_default * exposure_at_default * loss_given_default
    
    return round(expected_loss, 2)

if __name__ == "__main__":
    # Specify the dataset filename. 
    # Make sure you save the provided data as 'loan_data.csv' in the same folder.
    csv_file = 'loan_data.csv'
    
    # Train and compare models
    train_and_compare_models(csv_file)
    
    # If the model trained successfully, run a test calculation
    if best_model is not None:
        # Let's test with a sample borrower from the data
        # Example borrower with 0 default: 
        # 0 credit lines, $5221.54 loan, $3915.47 debt, $78039 income, 5 years employed, 605 FICO
        sample_borrower_good = {
            'credit_lines_outstanding': 0,
            'loan_amt_outstanding': 5221.54,
            'total_debt_outstanding': 3915.47,
            'income': 78039.38,
            'years_employed': 5,
            'fico_score': 605
        }
        
        # Example borrower with 1 default:
        # 5 credit lines, $1958.92 loan, $8228.75 debt, $26648 income, 2 years employed, 572 FICO
        sample_borrower_risky = {
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 1958.92,
            'total_debt_outstanding': 8228.75,
            'income': 26648.43,
            'years_employed': 2,
            'fico_score': 572
        }
        
        print("--- Expected Loss Estimations ---")
        
        loss_good = calculate_expected_loss(sample_borrower_good)
        print(f"Sample Good Borrower:")
        print(f"  Loan Amount: ${sample_borrower_good['loan_amt_outstanding']}")
        print(f"  Expected Loss: ${loss_good}")
        
        print()
        
        loss_risky = calculate_expected_loss(sample_borrower_risky)
        print(f"Sample Risky Borrower:")
        print(f"  Loan Amount: ${sample_borrower_risky['loan_amt_outstanding']}")
        print(f"  Expected Loss: ${loss_risky}")
