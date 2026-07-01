import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
import os

# Define the mathematical model
# x represents days since the first date in the dataset
# We use a fixed period of 365.25 days (1 year)
# Equation: y = m*x + c + A*sin((2*pi/365.25)*x + C)
def price_model(x, m, c, A, C):
    period = 365.25
    return m * x + c + A * np.sin((2 * np.pi / period) * x + C)

# Global variables to store model parameters and the reference date
_model_params = None
_first_date = None

def load_and_fit_model(csv_path: str = 'Nat_Gas.csv'):
    global _model_params, _first_date
    
    if not os.path.exists(csv_path):
        print(f"Warning: Data file '{csv_path}' not found. Cannot fit the model.")
        return
        
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Parse dates (assuming format like '10/31/2020' or '2020-10-31')
    # Use infer_datetime_format or simple to_datetime
    df['Dates'] = pd.to_datetime(df['Dates'])
    df = df.sort_values('Dates').reset_index(drop=True)
    
    # Set the first date as the reference point
    _first_date = df['Dates'].iloc[0]
    
    # Calculate days since the first date
    df['Days'] = (df['Dates'] - _first_date).dt.days
    
    # Extract x and y for fitting
    x_data = df['Days'].values
    y_data = df['Prices'].values
    
    # Initial guesses for the parameters [m, c, A, C]
    # m: slope (can be estimated by (y_last - y_first) / x_last)
    # c: intercept (approx y_first)
    # A: amplitude (approx (max - min) / 2)
    # C: phase shift (guess 0)
    m_guess = (y_data[-1] - y_data[0]) / (x_data[-1] if x_data[-1] != 0 else 1)
    c_guess = y_data[0]
    A_guess = (np.max(y_data) - np.min(y_data)) / 2
    C_guess = 0.0
    
    initial_guesses = [m_guess, c_guess, A_guess, C_guess]
    
    # Fit the curve
    popt, _ = curve_fit(price_model, x_data, y_data, p0=initial_guesses)
    
    _model_params = popt
    print("Model fitted successfully.")
    print(f"Fitted Parameters: m={popt[0]:.4f}, c={popt[1]:.4f}, A={popt[2]:.4f}, C={popt[3]:.4f}")
    
    return df

def estimate_price(date_input: str) -> float:
    """
    Estimates the price of natural gas for a given date using the fitted model.
    
    Args:
        date_input (str): The date string in 'YYYY-MM-DD' format (e.g., '2025-02-15').
        
    Returns:
        float: The estimated price rounded to two decimal places.
    """
    global _model_params, _first_date
    
    if _model_params is None or _first_date is None:
        raise ValueError("Model has not been fitted. Please load the data and fit the model first.")
        
    # Convert input string to datetime
    target_date = pd.to_datetime(date_input)
    
    # Calculate days since the reference date
    days_since = (target_date - _first_date).days
    
    # Calculate the price using the fitted model parameters
    estimated_price = price_model(days_since, *_model_params)
    
    return round(float(estimated_price), 2)

def plot_extrapolation(df, months_ahead=12):
    """
    Plots the original historical data and the model's fitted curve extending into the future.
    """
    if _model_params is None or _first_date is None:
        print("Cannot plot: Model has not been fitted.")
        return
        
    # Generate future dates for extrapolation
    last_date = df['Dates'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=months_ahead * 30, freq='D')
    
    # Combine historical and future dates for a smooth continuous line
    all_dates = pd.concat([df['Dates'], pd.Series(future_dates)]).drop_duplicates().sort_values()
    
    # Calculate days for all dates
    all_days = (all_dates - _first_date).dt.days.values
    
    # Calculate modeled prices
    modeled_prices = price_model(all_days, *_model_params)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.scatter(df['Dates'], df['Prices'], color='blue', label='Historical Data', zorder=5)
    
    # Plot model fit & extrapolation
    plt.plot(all_dates, modeled_prices, color='red', label='Fitted Model & Extrapolation', linewidth=2, zorder=4)
    
    # Add a vertical line to indicate where extrapolation begins
    plt.axvline(x=last_date, color='green', linestyle='--', label='Start of Extrapolation')
    
    plt.title('Natural Gas Price: Historical Data and Model Extrapolation')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path to the CSV file
    csv_filename = "Nat_Gas.csv"
    
    # Load data and fit the model
    df_historical = load_and_fit_model(csv_filename)
    
    if df_historical is not None:
        # Test the estimate_price function
        test_dates = ['2023-01-15', '2024-06-30', '2025-02-15', '2025-09-30']
        print("\nPrice Estimates:")
        for date_str in test_dates:
            try:
                price = estimate_price(date_str)
                print(f"Date: {date_str} -> Estimated Price: {price}")
            except Exception as e:
                print(f"Error estimating price for {date_str}: {e}")
                
        # Generate the visualization
        plot_extrapolation(df_historical, months_ahead=12)
