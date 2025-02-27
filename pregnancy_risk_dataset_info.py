"""
Pregnancy Risk Dataset Information

This file provides information about the maternal health risk dataset and how to use it
for training the pregnancy risk prediction model.

The maternal health risk dataset is available at:
https://github.com/Peco602/maternal-health-risk

Dataset Features:
- Age: Age in years when a woman is pregnant
- SystolicBP: Upper value of Blood Pressure in mmHg
- DiastolicBP: Lower value of Blood Pressure in mmHg
- BS: Blood glucose levels in mmol/L
- BodyTemp: Body temperature in degrees Celsius
- HeartRate: Heart rate in beats per minute
- RiskLevel: Target variable with three categories:
  - 0: low risk
  - 1: mid risk
  - 2: high risk

How to prepare the dataset:

1. Clone the repository:
   git clone https://github.com/Peco602/maternal-health-risk.git

2. Load and preprocess the dataset:
   - Handle missing values
   - Normalize/standardize features
   - Encode categorical variables if any
   - Split into training and testing sets

3. Use the PregnancyRiskPredictor.train() method with the preprocessed data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def download_and_prepare_dataset():
    """
    Downloads and prepares the maternal health risk dataset.
    This is a placeholder function - in a real implementation, you would:
    1. Download the dataset from the GitHub repository
    2. Preprocess the data
    3. Split into train/test sets
    """
    print("To download and prepare the maternal health risk dataset:")
    print("1. Clone the repository: git clone https://github.com/Peco602/maternal-health-risk.git")
    print("2. Use the following code to preprocess the dataset:")
    
    code_example = """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Load the dataset
    df = pd.read_csv('maternal-health-risk/data.csv')
    
    # Display basic information
    print(df.info())
    print(df.describe())
    
    # Check for missing values
    print(df.isnull().sum())
    
    # Encode the target variable
    le = LabelEncoder()
    df['RiskLevel'] = le.fit_transform(df['RiskLevel'])
    # 0: low risk, 1: mid risk, 2: high risk
    
    # Split features and target
    X = df.drop('RiskLevel', axis=1)
    y = df['RiskLevel']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Now the data is ready for training the model
    """
    
    print(code_example)

def explore_dataset():
    """
    Provides code for exploring and visualizing the maternal health risk dataset.
    """
    print("\nTo explore and visualize the dataset:")
    
    code_example = """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load the dataset
    df = pd.read_csv('maternal-health-risk/data.csv')
    
    # Map risk levels for better readability
    risk_map = {'low risk': 'Low Risk', 'mid risk': 'Medium Risk', 'high risk': 'High Risk'}
    df['RiskLevel'] = df['RiskLevel'].map(risk_map)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Distribution of risk levels
    plt.subplot(2, 3, 1)
    sns.countplot(x='RiskLevel', data=df)
    plt.title('Distribution of Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    
    # Age distribution by risk level
    plt.subplot(2, 3, 2)
    sns.boxplot(x='RiskLevel', y='Age', data=df)
    plt.title('Age Distribution by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Age (years)')
    
    # Blood pressure by risk level
    plt.subplot(2, 3, 3)
    sns.scatterplot(x='SystolicBP', y='DiastolicBP', hue='RiskLevel', data=df)
    plt.title('Blood Pressure by Risk Level')
    plt.xlabel('Systolic BP (mmHg)')
    plt.ylabel('Diastolic BP (mmHg)')
    
    # Blood glucose by risk level
    plt.subplot(2, 3, 4)
    sns.boxplot(x='RiskLevel', y='BS', data=df)
    plt.title('Blood Glucose by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Blood Glucose (mmol/L)')
    
    # Body temperature by risk level
    plt.subplot(2, 3, 5)
    sns.boxplot(x='RiskLevel', y='BodyTemp', data=df)
    plt.title('Body Temperature by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Body Temperature (°C)')
    
    # Heart rate by risk level
    plt.subplot(2, 3, 6)
    sns.boxplot(x='RiskLevel', y='HeartRate', data=df)
    plt.title('Heart Rate by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Heart Rate (bpm)')
    
    plt.tight_layout()
    plt.savefig('maternal_health_risk_exploration.png')
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()
    """
    
    print(code_example)

if __name__ == "__main__":
    download_and_prepare_dataset()
    explore_dataset()