# Week 4 — Credit Risk Assessment (Project Overview)

## Project Objective

Build an interpretable, well-documented credit risk scoring pipeline that estimates borrower risk using a proxy default label. The deliverables:

- Define and justify a proxy default target.
- Build and validate an interpretable baseline model (WoE + Logistic Regression).
- Compare with a higher-performance model (e.g., Gradient Boosting) and document trade-offs.
- Provide reproducible code, documentation, and validation artifacts suitable for regulatory review.

---

## Quick Highlights

- Emphasis on interpretability, documentation, and model governance.
- Reproducible experiments and clear evaluation against business metrics (PD, AUC, calibration).
- Recommendations for productionization and regulatory submission.

---

## Repository Structure

Suggested tree (top-level files/folders):

- .github/ — CI/workflow configs
- data/
  - raw/ — raw source files (not tracked)
  - processed/ — cleaned datasets used for modeling
- notebooks/
  - 01-eda.ipynb
- src/
  - data_manager.py
- scripts/
  - constants.py
  - decorator.py
- tests
  - test_data_manager.py
- .gitignore
- docker-compose.yml
- Dockerfile
- README.md
- requirements.txt

---

## Installation

1. Clone the repository:
   git clone git@github.com:Elshaday97/credit-risk-probability-week4.git
   cd credit-risk-probability-week4

2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate # macOS / Linux
   .venv\Scripts\activate # Windows

3. Install dependencies:
   pip install -r requirements.txt

---

## Data & Privacy Notes

- Raw data should live only in data/raw and not be committed.

---

## Usage

- Exploratory analysis:
  jupyter lab notebooks/01-eda.ipynb

---

## Modeling & Validation Guidance

- Proxy labeling: document definition, business rationale, and sensitivity analysis (vary the threshold).
- Feature engineering: use WoE binning for categorical and binned continuous variables to preserve interpretability.
- Baseline: Logistic Regression with WoE and monotonicity checks.
- Validation: AUC, KS, Brier score, calibration plots, PSI for population stability, and backtesting on temporal holdouts.
- Stress testing: scenario tests for macro shocks and feature drift.

---

## Regulatory & Documentation Checklist

- Model specification document (purpose, population, assumptions).
- Data lineage and feature descriptions.
- Training, validation, and backtest results with code notebooks.
- Implementation plan, monitoring strategy, and model risk mitigation measures.

---

## Credit Scoring Business Understanding

## Question 1

How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model? Why do Basel rules exist?

## Answer

## 1. Foundation: Why Regulation Exists

The need for regulation in risk management arose from issues inherent in bilateral contractual agreements and internal judgments. Credit risk has the potential to propagate through financial systems, causing "domino effects" and large-scale economic disruptions. High regulatory standards were introduced via the Basel Accords to mitigate this systemic risk.

## From Basel I to Basel II:

Basel I required banks to hold minimum capital against credit exposure based on standardized risk weights. For example, corporate loans had a 100% risk weight, while government loans had 0% (as gov't lending is safer than corporate lending). While this created structure, it was too simplistic and generalized.

Basel II was introduced to address these limitations. It established that credit risk is not just a matter of capital requirements, but also of internal governance, risk rating systems, and supervisory review.

## 2. The Three Pillars of Basel II

Basel II is supported by three key pillars that directly influence modeling choices:

**Pillar 1: Minimum Capital Requirements** This defines the rules for calculating capital requirements for credit, operational, and market risks.

Formula: Minimum Capital = Regulatory Capital ÷ Risk-Weighted Assets.

### Approaches: Banks can choose between:

**Standard Approach**: Regulators assign risk weights based on external credit ratings (simple but general).

**Internal Ratings-Based (IRB) Approach**: Banks build their own models to estimate key risk parameters: PD (Probability of Default), LGD (Loss Given Default), EAD (Exposure at Default), and Maturity. These parameters are combined to compute the required capital.

**Pillar 2: Supervisory Review** Regulators review a bank's internal processes to prevent capital from falling below the minimum requirement.

Banks must validate models, perform stress tests, and demonstrate that senior management understands the risks.

If a regulator believes the bank has underestimated risk (or cannot prove their model works), they can demand higher capital reserves.

**Pillar 3: Market Discipline** Transparency forces responsible behavior. Banks must publicly disclose their risk exposure, capital adequacy, and risk modeling methods to the market.

3. Connecting the dots: The Influence on Model Interpretability

These rules force Data Scientists to prioritize interpretability and documentation over pure predictive complexity.

The "White Box" Requirement: If a bank chooses the IRB Approach (Pillar 1), they are effectively asking the regulator to trust their internal math. To gain this trust, the model cannot be a "Black Box." It must be transparent regarding how inputs (like income or debt) influence the risk score.

Model Restrictions: This requirement effectively restricts the use of opaque algorithms (like deep Neural Networks) and favors interpretable models (like Logistic Regression or Decision Trees) where the logic is easily traceable.

Documentation & Audit: To satisfy Supervisory Review (Pillar 2), models must be rigorously documented. If a model is too complex to explain or document clearly, it will fail validation, forcing the bank back to the less favorable Standard Approach.

## Question 2

Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

## Answer

Creating a proxy variable is important to separate customers as high or low risk because raw data only contains behaviour not risk definition. ML models (specially supervised) require a labeled dataset to be trained. Our model must learn from this target variable, or else the algorithm will have no point of reference for low or high risk prediction. But defining this risk "proxy" incorrectly may go in to two extreme directions:

1. Extreme A (Too Strict): If we define the "default" as being 1 day late in payment, model might elimiate all possible loan profits
2. Extreme B (Too Loose): If we define the "default" as bankrupcy filing, then model thinks unless someone goes bankrupt, then they are low risk. This puts the banks in great danger of collapse.

## Question 3

What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The primary trade-off is between model accuracy in predicting the proxy variable and model interpretability.

1. Simple Models (Logistic Regression): These align well with regulatory requirements because they are transparent and easy to interpret. However, their shortcomings are that they assume linear relationships between variables and treat features independently. These assumptions can limit prediction accuracy, as financial data often contains outliers, skewed distributions, and complex patterns that a straight line cannot capture.

2. High-Performance Models (Gradient Boosting): These models perform better because they use Decision Trees . They handle curves and irregular patterns by creating many "if/then" splits, automatically capturing non-linear relationships and interactions between variables. On the flip side, they lack interpretability; it is difficult to pinpoint exactly why the model made a specific decision, making it challenging to validate for regulatory reporting.
