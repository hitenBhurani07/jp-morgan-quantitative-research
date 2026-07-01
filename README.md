# JP Morgan Quantitative Research Virtual Experience

## Overview
This repository contains my solutions and project files for the **JPMorgan Chase & Co. Quantitative Research** virtual experience program. The program simulates the day-to-day responsibilities of a Quantitative Researcher, ranging from pricing commodity contracts to analyzing retail banking credit risk. 

The project involves applying mathematical modeling, financial engineering, machine learning, and dynamic programming to solve complex, real-world business problems.

---

## 🚀 Task Breakdown & Methodology

### Task 1: Natural Gas Price Extrapolation and Modeling
**Objective:** Model and extrapolate historical natural gas prices to provide a granular, daily price estimate for past and future dates to assist the commodities trading desk.

**Workflow & Approach:**
- **Data Ingestion:** Parsed historical monthly natural gas price snapshots.
- **Mathematical Modeling:** Engineered a deterministic mathematical model combining a linear trend with a sinusoidal seasonality component.
  - *Equation:* `y = m*x + c + A * sin((2 * pi / 365.25) * x + C)`
  - *Why:* Natural gas exhibits long-term underlying inflation/trends and strict repeating 12-month seasonal cycles (peaking in winter, dropping in summer).
- **Optimization:** Utilized `scipy.optimize.curve_fit` to perfectly fit the parameters (`m`, `c`, `A`, `C`) against the historical data.
- **Output:** Built the `estimate_price(date_input)` function to accurately return expected prices rounded to two decimal places for any given date, alongside a predictive `matplotlib` visualization.
- **File:** `nat_gas_pricing.py`

### Task 2: Commodity Storage Contract Pricing
**Objective:** Develop a prototype pricing model for a natural gas storage contract that can handle multiple injections and withdrawals, factoring in various operational costs.

**Workflow & Approach:**
- **Cash Flow Analysis:** Approached the contract valuation strictly through the lens of expected cash flows: 
  - *Revenue:* Gas sold during withdrawal dates.
  - *Expenses:* Gas purchased during injection dates, fixed monthly storage fees, injection/withdrawal rate fees, and transport fees.
- **Validation Constraints:** Implemented strict logical checks to ensure volume limits and rate capacities were not exceeded, rejecting mathematically impossible scenarios (like over-withdrawing or exceeding maximum physical storage).
- **Integration:** Directly imported the predictive `estimate_price` model from Task 1 to simulate the market prices at the time of each injection/withdrawal.
- **File:** `contract_pricing.py`

### Task 3: Credit Risk Analysis (Probability of Default)
**Objective:** Assist the retail banking arm in building a predictive model to estimate the Probability of Default (PD) for personal loans and calculate the Expected Loss.

**Workflow & Approach:**
- **Machine Learning Selection:** Compared **Logistic Regression** and **Decision Tree Classifier** models to evaluate which algorithm generalized the loan data better based on the Area Under the ROC Curve (AUC).
- **Data Preprocessing:** Standardized numeric features (credit lines outstanding, loan amount, total debt, income, years employed, FICO score) using `StandardScaler` to ensure scale-invariant training.
- **Financial Risk Calculation:** After predicting the individual Probability of Default (PD), I applied the standard banking expected loss formula: 
  - *Formula:* `Expected Loss = PD * Exposure at Default (EAD) * Loss Given Default (LGD)`
  - *Assumption:* With a 10% recovery rate, LGD was strictly calculated as 90%.
- **File:** `credit_risk.py`

### Task 4: FICO Score Quantization (Bucketing)
**Objective:** Develop a robust, generalizable algorithm to strategically map continuous FICO scores into categorized buckets to optimize the predictive capability of the risk models.

**Workflow & Approach:**
- **Log-Likelihood Maximization:** Adopted a statistical approximation approach to maximize the log-likelihood of defaults in the generated buckets. The function considers the density of defaults and severely penalizes buckets with ambiguous default probabilities.
- **Dynamic Programming (DP):** Since exhaustively searching every possible FICO cutoff point is computationally unfeasible, I engineered a Dynamic Programming algorithm.
  - I precomputed the log-likelihood for every possible sub-range in $O(1)$ time.
  - The DP table dynamically built optimal combinations, mapping `b-1` buckets to sequentially find the absolute optimum limits for `b` buckets.
- **Impact:** Transformed a continuous integer scale (300-850) into a highly optimized, discrete set of categorical boundaries that the machine learning models (like in Task 3) can ingest seamlessly.
- **File:** `fico_bucketing.py`

---

## 💻 Tech Stack
- **Python 3.x**
- **Pandas & NumPy** (Data manipulation, aggregation, and mathematical operations)
- **Scikit-Learn** (Logistic Regression, Decision Trees, standard scaling, and accuracy metrics)
- **SciPy** (Mathematical curve fitting via Levenberg-Marquardt optimization)
- **Matplotlib** (Data visualization and curve projection)

## 🛠 How to Run
All tasks are compartmentalized into their respective standalone `.py` scripts. 
To run the models, ensure the dependent data files (`Nat_Gas.csv` and `loan_data.csv`) are located in the root directory alongside the scripts. 

Example execution:
```bash
python contract_pricing.py
python fico_bucketing.py
```
