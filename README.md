# Team 14 - LendingClub P2P Loan Default Prediction

**Course:** UE24CS352A - Machine Learning  
**Problem No.:** 39  
**Team:** Karan Varshney (PES2UG24CS903), D Sai Karthik (PES2UG24CS141)

## Problem statement
The project uses historical LendingClub loans to predict whether a finalized loan will be **Fully Paid** or will end in **Default / Charged Off**. The supplied reference paper also studies loan profitability, so the notebook includes an additional regression stage that predicts Net Annualized Return (NAR) and evaluates a simple loan-selection strategy.

## Main notebook
`Team14_LendingClub_Submission_Ready.ipynb`

The notebook contains:
- data download and filtering for finalized 2012-2015 loans
- missing-value analysis and EDA
- feature engineering and leakage prevention
- Logistic Regression, MLP and Random Forest classifiers
- validation-based decision-threshold selection
- confusion matrix, ROC/PR curves and feature importance
- Linear, Ridge, MLP and Random Forest regression
- Random Forest depth comparison (4, 8, 10)
- NAR-based loan-selection simulation
- live single-loan demonstration

## Run in Google Colab
1. Upload/open `Team14_LendingClub_Submission_Ready.ipynb` in Colab.
2. Use **Runtime -> Run all**.
3. The notebook downloads the LendingClub archive using `kagglehub`.
4. Keep `FAST_MODE = True` while testing.
5. For the final results, set `FAST_MODE = False`, restart the runtime and run all cells again.
6. Use the **Final review summary** and `demo_loan(0)` cells during the review.

## Dataset
The notebook uses the public LendingClub accepted-loan archive and keeps only finalized loans issued during 2012-2015. `Fully Paid` is mapped to class 1, while `Default` and `Charged Off` are mapped to class 0. Unfinished statuses such as `Current` and `Late` are excluded.

The current archived file produced 786,820 finalized rows in our initial run, with about 81.36% Fully Paid and 18.64% Default/Charged Off.

## Important implementation detail: leakage
`total_pymnt` and `last_pymnt_d` are not used as classifier inputs because they are known only after the loan is active. They are used later only to construct the NAR regression target.

## Models
### Classification
- Dummy majority baseline
- Logistic Regression with balanced class weights
- MLP neural network
- Random Forest

### Regression
- Linear Regression
- Ridge Regression
- MLP Regressor
- Random Forest Regressor with max depths 4, 8 and 10

## Evaluation
Classification is evaluated with default precision, recall, F1, weighted F1, ROC-AUC and PR-AUC. Regression is evaluated using MSE, RMSE and R2. Model and threshold selection are performed on the validation set, while the test set is used only for final evaluation.

## Suggested repository structure
```text
Team14-LendingClub/
├── Team14_LendingClub_Submission_Ready.ipynb
├── README.md
├── requirements.txt
├── report/
│   └── Team14_Writeup.pdf
├── presentation/
│   └── Team14_Presentation.pdf
└── results/
    ├── classification_results.csv
    ├── regression_results.csv
    ├── classification_thresholds.csv
    └── investment_strategy_validation.csv
```

## Live demonstration
After the notebook has been run, execute:

```python
demo_loan(0)
```

It displays a held-out loan, its actual outcome, predicted default probability, actual and predicted NAR, and the model-based investment decision.

## Team contribution
Before final submission, replace this section with the **actual contribution of each member**. The course evaluates individual contribution, so do not leave this section vague.

- **Karan Varshney:** [fill actual work]
- **D Sai Karthik:** [fill actual work]

## Reference
Peiqian Li and Gao Han, *LendingClub Loan Default and Profitability Prediction*, Stanford CS229 project.
