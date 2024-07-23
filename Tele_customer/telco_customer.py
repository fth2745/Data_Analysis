import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükleme
data = pd.read_csv('C:/Users/Fatih/Desktop/Data_Analysis_And_LinnerReg/Tele_customer/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

def customer_segmentation():
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].replace(' ', pd.NA))
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    features = data[['tenure', 'MonthlyCharges', 'TotalCharges']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Segment'] = kmeans.fit_predict(features_scaled)
    segment_churn_rates = data.groupby('Segment')['Churn'].mean()
    print(segment_churn_rates)
    print("-----------------------------------------")

def feature_importance():

    # Data preprocessing
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].replace(' ', pd.NA))
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

    # Convert categorical variables to numeric
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Feature importance
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df)
    print("-----------------------------------------")

def churn_rates_by_factors():
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    data.to_csv("vv.csv", index=False)
    demographic_factors = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    for factor in demographic_factors:
        print(f"Churn rate by {factor}:")
        print(data.groupby(factor)['Churn'].mean())
        print()
    service_factors = ['PhoneService', 'InternetService', 'Contract']
    for factor in service_factors:
        print(f"Churn rate by {factor}:")
        print(data.groupby(factor)['Churn'].mean())
        print()
        print("-----------------------------------------")

"""def time_series_analysis():

    # Zaman serisi analizi
    data['tenure_month'] = data['tenure'] * 30
    data['date'] = pd.to_datetime(data['tenure_month'], unit='D', origin='unix')
    ts_data = data.groupby('date').size().reset_index(name='churn_count')
    
    ts_data.rename(columns={'date': 'ds', 'churn_count': 'y'}, inplace=True)
    model = Prophet()
    model.fit(ts_data)
    
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    model.plot(forecast)
"""
def cohort_analysis():
    data['cohort_month'] = data['tenure'] // 12
    cohort_data = data.groupby(['cohort_month', 'tenure'])['Churn'].mean().unstack(0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cohort_data.T, cmap='coolwarm', annot=True)
    plt.title('Cohort Analysis')
    plt.show()
    print("-----------------------------------------")

def demographic_features_regression():
    X = data[['SeniorCitizen', 'tenure', 'MonthlyCharges']]
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Coefficient: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")
    print("-----------------------------------------")

def service_usage_regression():
    X = data[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Coefficient: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")
    print("-----------------------------------------")

def customer_satisfaction_regression():
    data['CustomerSatisfaction'] = data['MonthlyCharges'] / data['TotalCharges']
    X = data[['CustomerSatisfaction']]
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Coefficient: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")
    print("-----------------------------------------")

def contract_duration_regression():
    data['Contract_Months'] = data['Contract'].apply(lambda x: 24 if x == 'Two year' else 12 if x == 'One year' else 1)
    X = data[['Contract_Months']]
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Coefficient: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")
    print("-----------------------------------------")

def interaction_features_regression():
    data['Senior_tenure_interaction'] = data['SeniorCitizen'] * data['tenure']
    X = data[['SeniorCitizen', 'tenure', 'Senior_tenure_interaction']]
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Coefficient: {model.coef_}")
    print(f"Model Intercept: {model.intercept_}")
    print("-----------------------------------------")





customer_segmentation()
print("-----------------------------------------")
feature_importance()
print("-----------------------------------------")
churn_rates_by_factors()
print("-----------------------------------------")
cohort_analysis()
print("-----------------------------------------")
demographic_features_regression()
print("-----------------------------------------")
service_usage_regression()
print("-----------------------------------------")
customer_satisfaction_regression()
print("-----------------------------------------")
contract_duration_regression()
print("-----------------------------------------")
interaction_features_regression()
