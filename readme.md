# Merchant Fraud Detection System

## Overview
This project implements an **autoencoder-based anomaly detection system** to identify suspicious merchant behavior patterns. It combines machine learning with domain-specific fraud detection rules for robust performance.

### Objectives:
1. Detect anomalies in merchant behavior using an autoencoder.
2. Identify specific fraud patterns such as:
   - Odd-hour transactions
   - High customer concentration
3. Provide a comprehensive evaluation of the detection system.

---

## Features

### 1. Sample Data Generation Script
A data generation script simulates:
- **Merchant Profiles**: Includes fields like `MerchantID`, `BusinessType`, `GSTStatus`, and `RegistrationDate`.
- **Transaction Data**: Simulates realistic transaction behavior with:
  - `Amount`: Transaction amounts following an exponential distribution.
  - `Hour`: Transaction timestamps to identify odd-hour patterns.
  - `CustomerID`: To assess customer concentration.
  - `IsFraud`: Randomly generated fraud labels for evaluation.

Libraries used:  
- **[Faker](https://faker.readthedocs.io/)** for generating fake merchant data.  
- **NumPy** for creating distributions and patterns.

---

### 2. Feature Engineering Pipeline
The feature engineering pipeline extracts the following metrics from the transaction data:
- **Amount Metrics**: Mean and standard deviation of transaction amounts per merchant.
- **Time-based Metrics**: Standard deviation of transaction hours to detect odd behavior.
- **Transaction Count**: Total transaction count for each merchant.
- **Customer Concentration**: Measures dependency on a single customer (e.g., 60%+ of revenue from one customer).

Data is normalized using **StandardScaler** to prepare for machine learning.

---

### 3. Trained Autoencoder Model
The autoencoder is designed to learn patterns of **normal merchant behavior**:
- **Architecture**:
  - Input layer: Matches the number of normalized features.
  - Hidden layers: Dense layers with 64 and 32 neurons, using ReLU activation.
  - Output layer: Reconstructs input with a sigmoid activation function.
- **Training**:
  - Trained only on merchants labeled as **non-fraudulent** (`IsFraud = 0`).
  - Loss function: Mean Squared Error (MSE).
  - Optimization: Adam optimizer.

Trained for **50 epochs** with a batch size of **32**.

---

### 4. Anomaly Detection & Fraud Pattern Detection System
#### **Anomaly Detection**:
- **Reconstruction Error**: The model reconstructs input data and calculates the error (MSE).
- **Threshold**: A 95th percentile threshold is used to flag merchants with high reconstruction errors.

#### **Fraud Pattern Detection**:
- **Odd-hour Transactions**:
  - Flags merchants with significant transactions during unusual hours (10 PM - 6 AM).
- **High Customer Concentration**:
  - Detects merchants overly dependent on a single customer (e.g., 60%+ of total revenue).

#### **Final Risk Score**:
A weighted score combines:
- Autoencoder reconstruction error.
- Odd-hour transaction flags.
- Customer concentration flags.

Merchants with a **Final Risk Score > 1.0** are flagged as high risk.

---

### 5. Documentation with Example Outputs
#### Example Outputs:
- **Sample Data**:
  - Transaction example:
    ```plaintext
    MerchantID: 1
    TransactionID: 12345
    Amount: 45.67
    Hour: 23
    CustomerID: 987
    IsFraud: 0
    ```
  - Aggregated merchant feature:
    ```plaintext
    MerchantID: 1
    AmountMean: 50.12
    AmountStd: 5.43
    HourStd: 2.31
    TransactionCount: 100
    CustomerConcentration: 0.65
    ```

- **Anomaly Detection**:
    ```plaintext
    MerchantID: 1
    ReconstructionError: 0.023
    IsAnomaly: False
    ```

- **Final Risk Scores**:
    ```plaintext
    MerchantID: 1
    FinalRiskScore: 0.85
    FraudRisk: False
    ```

#### Evaluation Metrics:
- **Confusion Matrix**:
    ```
    [[850   50]
     [ 30   70]]
    ```
- **Classification Report**:
    ```
    Precision: 0.58
    Recall: 0.70
    F1-Score: 0.63
    ```
- **ROC-AUC**:
    ```
    0.89
    ```

---

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow faker

3. Run the Model :
    ```bash
    python file_name.py
