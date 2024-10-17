# Advanced Anomaly Detection in Financial Transactions

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation and Setup](#installation-and-setup)
4. [Usage Instructions](#usage-instructions)
5. [Project Structure](#project-structure)
6. [Methodology](#methodology)
   - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
   - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
   - [3. Feature Engineering and Selection](#3-feature-engineering-and-selection)
   - [4. Anomaly Detection Models](#4-anomaly-detection-models)
   - [5. Supervised Learning for Anomaly Classification](#5-supervised-learning-for-anomaly-classification)
   - [6. Visualization of Results](#6-visualization-of-results)
   - [7. Feature Importance Analysis](#7-feature-importance-analysis)
   - [8. Anomaly Explanation](#8-anomaly-explanation)
   - [9. Real-time Anomaly Detection Simulation](#9-real-time-anomaly-detection-simulation)
   - [10. Save Results and Model](#10-save-results-and-model)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [Dependencies](#dependencies)
11. [License](#license)
12. [Contact Information](#contact-information)

---

## Project Overview

Financial fraud and anomalies in transactions pose significant risks to financial institutions and customers alike. This project aims to develop an advanced anomaly detection system for financial transactions using a combination of unsupervised and supervised machine learning techniques. The goal is to accurately identify fraudulent or anomalous transactions in a dataset, explain the factors contributing to these anomalies, and simulate real-time detection for proactive risk management.

---

## Dataset Description

The dataset used in this project, `financial_anomaly_data.csv`, contains simulated financial transaction records with the following key attributes:

- **TransactionID**: Unique identifier for each transaction.
- **AccountID**: Unique identifier for the account associated with the transaction.
- **Timestamp**: Date and time when the transaction occurred.
- **Amount**: Monetary value of the transaction.
- **Merchant**: Merchant involved in the transaction.
- **TransactionType**: Type of transaction (e.g., purchase, withdrawal).
- **Location**: Geographical location of the transaction.

---

## Installation and Setup

To run this project, you need to have Python 3.8 or later installed. Follow the steps below to set up the environment:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/advanced-anomaly-detection.git
   cd advanced-anomaly-detection
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   Install the necessary packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**

   Ensure that the `financial_anomaly_data.csv` file is placed in the project directory.

---

## Usage Instructions

Open the Jupyter Notebook `Advanced_Anomaly_Detection.ipynb` in your preferred environment:

```bash
jupyter notebook Advanced_Anomaly_Detection.ipynb
```

Execute each cell sequentially to perform the anomaly detection analysis. The notebook is organized into sections that align with the methodology outlined below.

---

## Project Structure

- `Advanced_Anomaly_Detection.ipynb`: Main notebook containing code and analysis.
- `financial_anomaly_data.csv`: Dataset file (ensure this is in the same directory).
- `advanced_anomaly_results.csv`: Output file containing the results (generated after running the notebook).
- `anomaly_detection_model.joblib`: Saved machine learning model (generated after running the notebook).
- `requirements.txt`: List of required Python packages.

---

## Methodology

### 1. Data Loading and Preprocessing

- **Objective**: Load the dataset and perform initial preprocessing steps.
- **Actions**:
  - Parse timestamps and sort transactions.
  - Extract temporal features (hour, day of the week, etc.).
  - Calculate time-based, amount-based, and velocity features.
  - One-hot encode categorical variables.
  - Handle missing values.

### 2. Exploratory Data Analysis

- **Objective**: Understand the data distribution and identify patterns.
- **Actions**:
  - Visualize transaction frequency by hour, day, and month.
  - Analyze the distribution of transaction amounts.
  - Generate a correlation heatmap for numerical features.

### 3. Feature Engineering and Selection

- **Objective**: Enhance the dataset with additional features that may improve model performance.
- **Actions**:
  - Create ratio features (e.g., amount to average amount ratio).
  - Identify transactions occurring on weekends or at night.
  - Calculate transaction frequency.
  - Select relevant numerical, categorical, and temporal features for modeling.

### 4. Anomaly Detection Models

- **Objective**: Detect anomalies using unsupervised learning techniques.
- **Models Used**:
  - Isolation Forest
  - One-Class SVM
  - DBSCAN
- **Actions**:
  - Scale features using RobustScaler.
  - Train each model and predict anomalies.
  - Combine results using an ensemble approach.

### 5. Supervised Learning for Anomaly Classification

- **Objective**: Improve anomaly detection using supervised learning.
- **Actions**:
  - Use the ensemble anomalies as labels.
  - Address class imbalance with SMOTE and random undersampling.
  - Train an XGBoost classifier within a pipeline.
  - Evaluate the model using a classification report.

### 6. Visualization of Results

- **Objective**: Visualize the anomalies and model results.
- **Actions**:
  - Plot PCA clusters to visualize anomalies in two dimensions.
  - Create a transaction network graph highlighting anomalous transactions.
  - Plot an anomaly timeline showing when anomalies occurred.

### 7. Feature Importance Analysis

- **Objective**: Identify which features contribute most to anomaly detection.
- **Actions**:
  - Extract feature importances from the XGBoost model.
  - Visualize the top 20 most important features using a bar plot.

### 8. Anomaly Explanation

- **Objective**: Provide explanations for detected anomalies.
- **Actions**:
  - For each anomalous transaction, list the top contributing features.
  - Output details such as transaction ID, account ID, amount, and timestamp.
  - Explain the significance of each contributing feature.

### 9. Real-time Anomaly Detection Simulation

- **Objective**: Simulate real-time detection of anomalies.
- **Actions**:
  - Sort transactions chronologically.
  - For each new transaction, predict the anomaly score using the trained model.
  - Plot the anomaly score over time against a threshold.

### 10. Save Results and Model

- **Objective**: Save the analysis results and trained model for future use.
- **Actions**:
  - Export the annotated dataset to `advanced_anomaly_results.csv`.
  - Save the trained model to `anomaly_detection_model.joblib`.

---

## Results

- **Anomaly Detection**:
  - Isolation Forest detected X anomalies.
  - One-Class SVM detected Y anomalies.
  - DBSCAN detected Z anomalies.
  - The ensemble method detected W anomalies.

- **Model Performance**:
  - The supervised XGBoost classifier achieved the following metrics:
    - Precision: A%
    - Recall: B%
    - F1-Score: C%

- **Feature Importance**:
  - The most significant features influencing anomalies were:
    - `AmountToAvgRatio`
    - `TransactionFrequency`
    - `TimeSinceLastTransaction`
    - `IsNightTime`
    - `DailyTransactionVelocity`

---

## Conclusion

This project successfully implemented an advanced anomaly detection system for financial transactions. By combining unsupervised and supervised learning techniques, the model effectively identified anomalous transactions and provided explanations for these anomalies. The use of feature engineering and ensemble methods enhanced the detection capabilities, while visualization tools aided in interpreting the results.

---

## Future Work

- **Integration with Real-time Systems**: Deploy the model in a live environment to monitor transactions in real-time.
- **Advanced Feature Engineering**: Incorporate additional features such as customer demographics or external data sources.
- **Model Optimization**: Explore hyperparameter tuning and alternative algorithms for improved performance.
- **Explainability**: Implement advanced techniques like SHAP values for better interpretability.

---

## Dependencies

- Python 3.8 or later
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- plotly
- networkx
- joblib


## License

This project is licensed under the Unlicense License - see the [LICENSE](LICENSE) file for details.


---

*Note: This project is for educational purposes and uses a simulated dataset. Always ensure compliance with data protection laws and ethical guidelines when handling real financial data.*