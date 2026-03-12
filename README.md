# Linear Regression with Gradient Descent (MATLAB)
<img width="1469" height="1068" alt="Figure_2 (1)" src="https://github.com/user-attachments/assets/afb2e34b-3e62-4a18-b78a-892ba9c17dfc"/> 
![MATLAB](https://img.shields.io/badge/MATLAB-Numerical%20Computing-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-blue)
![Optimization](https://img.shields.io/badge/Optimization-Gradient%20Descent-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

🔗 Live Demo:
https://mustafa0783.github.io/linear-regression-gradient-descent/

A MATLAB project implementing **Linear Regression using Gradient Descent** and comparing **analytic gradients with numerical gradient approximations**.

This project demonstrates how gradient-based optimization converges to optimal parameters and evaluates performance using both **synthetic datasets** and **real housing price data**.

---
## Tech Stack

• MATLAB  
• Machine Learning Algorithms  
• Gradient Descent Optimization  
• Data Visualization  
• Numerical Methods


# Project Highlights

• Linear Regression implemented **from scratch**
• Comparison of **3 gradient descent approaches**

* Analytic Gradient Descent
* Numerical Gradient Descent (Forward Difference)
* Numerical Gradient Descent (Central Difference)

• Visualization of:

* learning curves
* regression fitting
* gradient error
* residual analysis
* parameter convergence

---

# Mathematical Model

Linear Regression models the relationship between an input feature (x) and output (y):

```
y = mx + b
```

Where

* **m** = slope (weight)
* **b** = intercept (bias)

The model minimizes **Mean Squared Error (MSE)**

```
MSE = (1/n) Σ (y - y_pred)^2
```

Gradient descent updates parameters iteratively

```
m = m - α * ∂MSE/∂m
b = b - α * ∂MSE/∂b
```

where **α** is the learning rate.

---

# Project Workflow

The project consists of two main experiments.

---

# Part 1 — Synthetic Data Experiment

Synthetic data was generated using

```
y = 2.5x + 5
```

This verifies whether gradient descent can recover the true parameters.

### Parameter Comparison

| Method               | Slope (m) | Intercept (b) | Final MSE |
| -------------------- | --------- | ------------- | --------- |
| True                 | 2.50      | 5.00          | -         |
| Analytic GD          | 2.4472    | 5.4601        | 0.075933  |
| Numeric GD (Forward) | 2.4707    | 5.3275        | 0.075933  |
| Numeric GD (Central) | 2.4480    | 5.4599        | 0.075933  |

All approaches converge close to the true values.

---

# Synthetic Data Regression

<img width="484" height="532" alt="Synthetic Data Regression" src="https://github.com/user-attachments/assets/5a61dfe2-ce8b-46fb-871d-fbff5ba2bd0d" />


---

# Learning Curves (MSE vs Iterations)

<img width="501" height="531" alt="MSE vs Iterations" src="https://github.com/user-attachments/assets/f49a0da8-eb5a-45e8-9b49-ce5d27ff79c1" />


---

# Gradient Approximation Error

<img width="495" height="532" alt="Gradient Approximation Error" src="https://github.com/user-attachments/assets/a83e8237-6f18-4555-9387-465aaa38821e" />


---

# Part 2 — Real Dataset: House Price Prediction

Dataset contains

Feature:

```
Square Footage
```

Target:

```
House Price
```

Dataset statistics

Total samples: **199**

Training samples: **159**

Testing samples: **40**

Price range:

```
$146,406 – $1,107,045
```

---

# Training Data with Regression Fit

<img width="492" height="517" alt="Training Data with Regression Fit" src="https://github.com/user-attachments/assets/71b4bef4-93a9-415c-a51c-6c43819dafaa" />

---

# Predicted vs Actual Prices


<img width="523" height="519" alt="Figure_2" src="https://github.com/user-attachments/assets/31187745-e709-4571-bd36-a743baa506d8" />

---

# Residual Analysis


<img width="476" height="538" alt="Residual Analysis" src="https://github.com/user-attachments/assets/479203d5-928a-4558-8914-36de37064214" />

---

# Parameter Convergence

<img width="548" height="514" alt="Parameter Convergence" src="https://github.com/user-attachments/assets/82ec1525-65fe-4866-a0df-e42d9beb9605" />


---

# Error Distribution

<img width="436" height="533" alt="Error Distribution" src="https://github.com/user-attachments/assets/85f18562-9b67-41bd-9068-69f86c1a6ea0" />


---

# Model Performance

### Analytic Gradient Descent

Slope:

```
m = 199.44
```

Intercept:

```
b = 59826.48
```

Regression Equation

```
Price = 199.44 * Square_Footage + 59826.48
```

### Evaluation Metrics

| Metric        | Analytic GD | Numeric GD |
| ------------- | ----------- | ---------- |
| RMSE          | $31,095     | $31,601    |
| R² Score      | 0.9823      | 0.9817     |
| Training Time | 0.0008 s    | 0.0016 s   |

Interpretation

• Each additional square foot increases house price by about **$199**
• Model explains **98.23% of price variation**

---

# Key Insights

Analytic gradient descent converges faster than numerical gradient methods.

Central difference gradient approximation is more accurate than forward difference.

The regression model performs very well on real housing data with **high R² score**.

Residual plots show no major systematic errors.

---

## Project Structure

main.m              -> Main script to run the project  
data/               -> Housing dataset  
outputs/            -> Generated plots and visualizations  

# How to Run the Project

### Requirements

MATLAB R2020 or newer

---

### Run

Open MATLAB and execute

```
main.m
```

All generated plots will be saved to


============================================
Numerical Computations Lab Project - Fall 2025
Linear Regression with Gradient Descent
============================================

PART 1: Testing on Synthetic Data
---------------------------------
Generating synthetic data...
True relationship: y = 2.50 * X + 5.00
Data shape: 100 samples

Training with Analytic Gradient Descent...
  Final m: 2.4472 (True: 2.50)
  Final b: 5.4601 (True: 5.00)
  Final MSE: 0.075933
  Iterations: 256
  Time: 0.0025 seconds

Training with Numerical Gradient Descent (Forward Difference)...
  Final m: 2.4707
  Final b: 5.3275
  Final MSE: 0.075933
  Iterations: 299
  Time: 0.0041 seconds

Training with Numerical Gradient Descent (Central Difference)...
  Final m: 2.4480
  Final b: 5.4599
  Final MSE: 0.075933
  Iterations: 283
  Time: 0.0037 seconds

Parameter Comparison:
-------------------------------------------------
Method                 m          b          MSE
-------------------------------------------------
True                  2.5000     5.0000       -
Analytic GD           2.4472     5.4601   0.075933
Numeric GD (Fwd)      2.4707     5.3275   0.075933
Numeric GD (Cent)     2.4480     5.4599   0.075933
-------------------------------------------------


PART 2: Training on Real Data
---------------------------------
Loading real dataset...
Dataset loaded: 199 samples
Features: Square_Footage vs House_Price
Price range: $146406.90 to $1107045.06

Data split:
  Training samples: 159
  Test samples: 40

Training with Analytic Gradient Descent...
  Final slope (m): 199.4404
  Final intercept (b): 59826.4812
  Final MSE: 0.020753
  Iterations: 251
  Time: 0.0008 seconds

Training with Numerical Gradient Descent (Central Difference)...
  Final slope (m): 198.0538
  Final intercept (b): 62108.1342
  Final MSE: 0.020753
  Iterations: 329
  Time: 0.0016 seconds

Performance on Test Set:
------------------------------------------------------------
Metric                Analytic GD      Numeric GD (Cent)
------------------------------------------------------------
RMSE                $    31095.21   $    31601.00
R-squared                 0.9823          0.9817
Training Time (s)         0.0008          0.0016
------------------------------------------------------------

Regression Equations:
Analytic GD: Price = 199.44 * Square_Footage + 59826.48
Numeric GD:  Price = 198.05 * Square_Footage + 62108.13

Interpretation:
• Each additional square foot increases house price by approximately $199.44
• The base price (for 0 sq ft) is $59826.48
• R² = 0.9823 means 98.23% of price variation is explained by square footage


============================================
Project Execution Complete
All plots saved to /outputs/ folder
============================================
>> 

---

## Author

**Mustafa Dawood**  
BS Computer Science  
Pak-Austria Fachhochschule Institute of Applied Sciences and Technology (PAF-IAST)

GitHub: https://github.com/mustafa0783
LinkedIn: https://www.linkedin.com/in/mustafadawood121/

# Optional Future Work

Possible improvements

* Polynomial Regression
* Multivariate Regression
* Stochastic Gradient Descent
* Feature normalization
* Regularization (Ridge / Lasso)

