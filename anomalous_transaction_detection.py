# Advanced Anomaly Detection in Financial Transactions
# Author: Claude
# Date: October 16, 2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import networkx as nx
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# 1. Data Loading and Preprocessing
print("1. Data Loading and Preprocessing")

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
    data = data.sort_values(by=['AccountID', 'Timestamp'])
    
    # Extract temporal features
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month
    data['DayOfMonth'] = data['Timestamp'].dt.day
    data['WeekOfYear'] = data['Timestamp'].dt.isocalendar().week
    
    # Calculate time-based features
    data['TimeSinceLastTransaction'] = data.groupby('AccountID')['Timestamp'].diff().dt.total_seconds()
    data['TimeSinceLastTransaction'].fillna(data['TimeSinceLastTransaction'].median(), inplace=True)
    
    # Calculate transaction statistics
    data['DailyTransactionCount'] = data.groupby(['AccountID', data['Timestamp'].dt.date])['TransactionID'].transform('count')
    data['WeeklyTransactionCount'] = data.groupby(['AccountID', data['Timestamp'].dt.to_period('W')])['TransactionID'].transform('count')
    data['MonthlyTransactionCount'] = data.groupby(['AccountID', data['Timestamp'].dt.to_period('M')])['TransactionID'].transform('count')
    
    # Calculate amount-based features
    data['AverageTransactionAmount'] = data.groupby('AccountID')['Amount'].transform('mean')
    data['TransactionAmountDeviation'] = data['Amount'] - data['AverageTransactionAmount']
    data['DailyTotalAmount'] = data.groupby(['AccountID', data['Timestamp'].dt.date])['Amount'].transform('sum')
    data['WeeklyTotalAmount'] = data.groupby(['AccountID', data['Timestamp'].dt.to_period('W')])['Amount'].transform('sum')
    data['MonthlyTotalAmount'] = data.groupby(['AccountID', data['Timestamp'].dt.to_period('M')])['Amount'].transform('sum')
    
    # Calculate velocity features
    data['DailyTransactionVelocity'] = data['DailyTransactionCount'] / 24
    data['WeeklyTransactionVelocity'] = data['WeeklyTransactionCount'] / 168
    data['MonthlyTransactionVelocity'] = data['MonthlyTransactionCount'] / (30 * 24)
    
    # One-Hot Encode categorical variables
    categorical_cols = ['Merchant', 'TransactionType', 'Location']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    data = data.dropna(subset=['AccountID', 'Timestamp'])  # Drop rows with NaN in critical columns
    return data

data = load_and_preprocess_data('financial_anomaly_data.csv')
print(data.head())
print(data.info())

# 2. Exploratory Data Analysis
print("\n2. Exploratory Data Analysis")

def plot_transaction_patterns(data):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Transactions by Hour", "Transactions by Day of Week", 
                                                        "Transactions by Month", "Amount Distribution"))
    
    # Transactions by Hour
    hourly_transactions = data.groupby('Hour').size()
    fig.add_trace(go.Bar(x=hourly_transactions.index, y=hourly_transactions.values, name="Hourly"), row=1, col=1)
    
    # Transactions by Day of Week
    daily_transactions = data.groupby('DayOfWeek').size()
    fig.add_trace(go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=daily_transactions.values, name="Daily"), row=1, col=2)
    
    # Transactions by Month
    monthly_transactions = data.groupby('Month').size()
    fig.add_trace(go.Bar(x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                         y=monthly_transactions.values, name="Monthly"), row=2, col=1)
    
    # Amount Distribution
    fig.add_trace(go.Histogram(x=data['Amount'], name="Amount"), row=2, col=2)
    
    fig.update_layout(height=800, width=1000, title_text="Transaction Patterns")
    fig.show()

plot_transaction_patterns(data)

def plot_correlation_heatmap(data):
    numerical_cols = ['Amount', 'TimeSinceLastTransaction', 'AverageTransactionAmount', 'TransactionAmountDeviation',
                      'DailyTransactionCount', 'WeeklyTransactionCount', 'MonthlyTransactionCount',
                      'DailyTotalAmount', 'WeeklyTotalAmount', 'MonthlyTotalAmount',
                      'DailyTransactionVelocity', 'WeeklyTransactionVelocity', 'MonthlyTransactionVelocity']
    
    corr_matrix = data[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

plot_correlation_heatmap(data)

# 3. Feature Engineering and Selection
print("\n3. Feature Engineering and Selection")

def engineer_features(data):
    # Ratio features
    data['AmountToAvgRatio'] = data['Amount'] / data['AverageTransactionAmount']
    data['DailyToWeeklyCountRatio'] = data['DailyTransactionCount'] / data['WeeklyTransactionCount']
    data['WeeklyToMonthlyCountRatio'] = data['WeeklyTransactionCount'] / data['MonthlyTransactionCount']
    
    # Time-based features
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    data['IsNightTime'] = ((data['Hour'] >= 22) | (data['Hour'] < 6)).astype(int)
    
    # Amount percentiles
    data['AmountPercentile'] = data.groupby('AccountID')['Amount'].rank(pct=True)
    
    # Transaction frequency features
    data['TransactionFrequency'] = data.groupby('AccountID')['TransactionID'].transform('count') / \
                                   (data.groupby('AccountID')['Timestamp'].transform('max') - 
                                    data.groupby('AccountID')['Timestamp'].transform('min')).dt.total_seconds()
    
    return data

data = engineer_features(data)

def select_features(data):
    numerical_cols = ['Amount', 'TimeSinceLastTransaction', 'AverageTransactionAmount', 'TransactionAmountDeviation',
                      'DailyTransactionCount', 'WeeklyTransactionCount', 'MonthlyTransactionCount',
                      'DailyTotalAmount', 'WeeklyTotalAmount', 'MonthlyTotalAmount',
                      'DailyTransactionVelocity', 'WeeklyTransactionVelocity', 'MonthlyTransactionVelocity',
                      'AmountToAvgRatio', 'DailyToWeeklyCountRatio', 'WeeklyToMonthlyCountRatio',
                      'AmountPercentile', 'TransactionFrequency']
    
    categorical_cols = [col for col in data.columns if col.startswith(('Merchant_', 'TransactionType_', 'Location_'))]
    
    temporal_cols = ['Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'WeekOfYear', 'IsWeekend', 'IsNightTime']
    
    feature_cols = numerical_cols + categorical_cols + temporal_cols
    
    return data[feature_cols]

X = select_features(data)
print("Selected features:", X.columns.tolist())

# 4. Anomaly Detection Models
print("\n4. Anomaly Detection Models")

def train_isolation_forest(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    anomalies = isolation_forest.fit_predict(X_scaled)
    return anomalies == -1

def train_one_class_svm(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    one_class_svm = OneClassSVM(kernel='rbf', nu=0.01)
    anomalies = one_class_svm.fit_predict(X_scaled)
    return anomalies == -1

def train_dbscan(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)
    return clusters == -1

# Train models
if_anomalies = train_isolation_forest(X)
ocsvm_anomalies = train_one_class_svm(X)
dbscan_anomalies = train_dbscan(X)

# Combine results
data['IF_Anomaly'] = if_anomalies
data['OCSVM_Anomaly'] = ocsvm_anomalies
data['DBSCAN_Anomaly'] = dbscan_anomalies
data['Ensemble_Anomaly'] = ((if_anomalies.astype(int) + ocsvm_anomalies.astype(int) + dbscan_anomalies.astype(int)) >= 2).astype(int)

print("Anomalies detected:")
print(f"Isolation Forest: {if_anomalies.sum()}")
print(f"One-Class SVM: {ocsvm_anomalies.sum()}")
print(f"DBSCAN: {dbscan_anomalies.sum()}")
print(f"Ensemble: {data['Ensemble_Anomaly'].sum()}")

# 5. Supervised Learning for Anomaly Classification
print("\n5. Supervised Learning for Anomaly Classification")

def train_supervised_model(X, y):
    # Convert y to integer type and X to float type
    y = y.astype(int)
    X = X.astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define the resampling strategy
    over = SMOTE(sampling_strategy=0.1, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    
    # Create a pipeline with SMOTE, undersampling, and XGBoost
    model = Pipeline([
        ('over', over),
        ('under', under),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    return model

# Train supervised model using ensemble anomalies as labels
supervised_model = train_supervised_model(X, data['Ensemble_Anomaly'])

# 6. Visualization of Results
print("\n6. Visualization of Results")

def plot_pca_clusters(X, anomalies, title):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], c='blue', alpha=0.5, label='Normal')
    plt.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', alpha=0.5, label='Anomaly')
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.show()

plot_pca_clusters(X, data['Ensemble_Anomaly'], 'PCA Visualization of Anomalies')

def plot_transaction_network(data):
    G = nx.Graph()
    
    # Check if 'Merchant' column exists, if not, use 'TransactionType' or another appropriate column
    if 'Merchant' in data.columns:
        merchant_col = 'Merchant'
    elif 'TransactionType' in data.columns:
        merchant_col = 'TransactionType'
    else:
        print("Neither 'Merchant' nor 'TransactionType' column found. Unable to create transaction network.")
        return
    
    # Add nodes
    for account in data['AccountID'].unique():
        G.add_node(account, node_type='account')
    for merchant in data[merchant_col].unique():
        G.add_node(merchant, node_type='merchant')
    
    # Add edges
    for _, transaction in data.iterrows():
        G.add_edge(transaction['AccountID'], transaction[merchant_col], weight=transaction['Amount'])
    
    # Set node colors
    node_colors = ['lightblue' if G.nodes[node]['node_type'] == 'account' else 'lightgreen' for node in G.nodes()]
    
    # Set node sizes
    node_sizes = [100 if G.nodes[node]['node_type'] == 'account' else 50 for node in G.nodes()]
    
    # Plot the graph
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=True, font_size=8, edge_color='gray', alpha=0.6)
    
    # Highlight anomalous transactions
    anomalous_edges = [(row['AccountID'], row[merchant_col]) for _, row in data[data['Ensemble_Anomaly'] == 1].iterrows()]
    nx.draw_networkx_edges(G, pos, edgelist=anomalous_edges, edge_color='red', width=2)
    
    plt.title(f'Transaction Network using {merchant_col} (Red edges indicate anomalies)')
    plt.axis('off')
    plt.show()

# Call the function
plot_transaction_network(data)

def plot_anomaly_timeline(data):
    fig = go.Figure()
    
    # Plot all transactions
    fig.add_trace(go.Scatter(x=data['Timestamp'], y=data['Amount'],
                             mode='markers', name='Normal Transactions',
                             marker=dict(color='blue', size=5, opacity=0.5)))
    
    # Plot anomalous transactions
    anomalies = data[data['Ensemble_Anomaly'] == 1]
    fig.add_trace(go.Scatter(x=anomalies['Timestamp'], y=anomalies['Amount'],
                             mode='markers', name='Anomalous Transactions',
                             marker=dict(color='red', size=8, symbol='star')))
    
    fig.update_layout(title='Transaction Timeline with Anomalies',
                      xaxis_title='Timestamp',
                      yaxis_title='Transaction Amount',
                      showlegend=True)
    fig.show()

plot_anomaly_timeline(data)

# 7. Feature Importance Analysis
print("\n7. Feature Importance Analysis")

def plot_feature_importance(model, X):
    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.show()

plot_feature_importance(supervised_model, X)

# 8. Anomaly Explanation
print("\n8. Anomaly Explanation")

def explain_anomalies(data, X, model):
    anomalies = data[data['Ensemble_Anomaly'] == 1]
    X_anomalies = X.loc[anomalies.index]
    
    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_names = X.columns
    
    for idx, (_, anomaly) in enumerate(anomalies.iterrows(), 1):
        print(f"\nAnomalous Transaction {idx}:")
        print(f"Transaction ID: {anomaly['TransactionID']}")
        print(f"Account ID: {anomaly['AccountID']}")
        print(f"Amount: ${anomaly['Amount']:.2f}")
        print(f"Timestamp: {anomaly['Timestamp']}")
        
        # Get feature values for this anomaly
        feature_values = X_anomalies.loc[anomaly.name]
        
        # Sort features by importance and get top 5
        top_features = sorted(zip(feature_names, feature_importance, feature_values), 
                              key=lambda x: x[1], reverse=True)[:5]
        
        print("Top contributing factors:")
        for feature, importance, value in top_features:
            print(f"- {feature}: {value:.2f} (importance: {importance:.4f})")
        
        print("-" * 50)

explain_anomalies(data, X, supervised_model)

# 9. Real-time Anomaly Detection Simulation
print("\n9. Real-time Anomaly Detection Simulation")

def simulate_real_time_detection(data, model, window_size=1000):
    # Sort data by timestamp
    data_sorted = data.sort_values('Timestamp')
    
    # Initialize lists to store results
    timestamps = []
    anomaly_scores = []
    
    # Simulate real-time detection
    for i in range(window_size, len(data_sorted)):
        window = data_sorted.iloc[i-window_size:i]
        current_transaction = data_sorted.iloc[i]
        
        # Prepare features for the current transaction
        X_current = select_features(pd.DataFrame([current_transaction]))
        
        # Predict anomaly probability
        anomaly_prob = model.predict_proba(X_current)[0, 1]
        
        timestamps.append(current_transaction['Timestamp'])
        anomaly_scores.append(anomaly_prob)
    
    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=anomaly_scores, mode='lines', name='Anomaly Score'))
    fig.add_trace(go.Scatter(x=timestamps, y=[0.5]*len(timestamps), mode='lines', name='Threshold', line=dict(dash='dash')))
    
    fig.update_layout(title='Real-time Anomaly Detection Simulation',
                      xaxis_title='Timestamp',
                      yaxis_title='Anomaly Score',
                      showlegend=True)
    fig.show()

simulate_real_time_detection(data, supervised_model)

# 10. Save Results and Model
print("\n10. Save Results and Model")

# Save results
data.to_csv('advanced_anomaly_results.csv', index=False)
print("Results saved to 'advanced_anomaly_results.csv'")

# Save model
import joblib
joblib.dump(supervised_model, 'anomaly_detection_model.joblib')
print("Model saved to 'anomaly_detection_model.joblib'")

print("\nAdvanced Anomaly Detection analysis completed successfully!")