import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Veri setini okuyoruz
df = pd.read_csv("./vine_quality/winequality-red.csv")

# Veri setindeki eksik verileri kontrol ediyoruz
print(df.isnull().sum())  # Her bir sütundaki eksik verilerin sayısını yazdırır

# Veri setinin istatistiksel özetini alıyoruz
print(df.describe())  # Her bir sütunun istatistiksel özetini yazdırır

# Histogramları çizmek için fonksiyon
def hist():
    sea.histplot(data=df, color="b")  # Verileri histogram şeklinde görselleştiriyoruz
    plt.show()

# Korelasyon matrisini görselleştirmek için fonksiyon
def corre(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Üçgen maske oluşturuyoruz
    sea.heatmap(corr, mask=mask, cmap="YlGnBu", annot=True, fmt=".2f")  # Isı haritası çiziyoruz
    plt.show()

# Scatter plotları çizmek için fonksiyon
def scatter(df):
    corr = df.corr()
    top_corr_pairs = corr.unstack().sort_values(ascending=False).drop_duplicates().head(10)  # En yüksek korelasyonlu çiftleri alıyoruz
    
    plt.figure(figsize=(12, 10))
    for i, (var1, var2) in enumerate(top_corr_pairs.index):
        if var1 != var2:
            plt.subplot(5, 2, i + 1)
            sea.scatterplot(x=df[var1], y=df[var2])  # Scatter plot çiziyoruz
            plt.title(f'{var1} vs {var2}')
        
    plt.tight_layout()
    plt.show()

# Çiftler arasındaki ilişkileri incelemek için fonksiyon
def pair(df):
    sea.pairplot(df, hue="quality", vars=df.columns.drop("quality"))  # Pair plot çiziyoruz
    plt.show()

# Veriyi hazırlama ve modelleme işlemleri
def data_manipulate(df):
    # Split data into training and testing sets
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # Principal Component Analysis (PCA)
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train models on RFE and PCA features
    model_rfe = LinearRegression()
    model_rfe.fit(X_train_rfe, y_train)
    y_pred_rfe = model_rfe.predict(X_test_rfe)

    model_pca = LinearRegression()
    model_pca.fit(X_train_pca, y_train)
    y_pred_pca = model_pca.predict(X_test_pca)

    # Evaluate model performance
    results_rfe = {
        'Model': ['RFE'],
        'MSE': [mean_squared_error(y_test, y_pred_rfe)],
        'R2': [r2_score(y_test, y_pred_rfe)]
    }

    results_pca = {
        'Model': ['PCA'],
        'MSE': [mean_squared_error(y_test, y_pred_pca)],
        'R2': [r2_score(y_test, y_pred_pca)]
    }

    # Plot model performance
    def plot_model_performance(results, title_prefix):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title_prefix} Model Performance Comparison')

        sea.barplot(x="Model", y="MSE", data=pd.DataFrame(results), ax=axes[0])
        axes[0].set_title(f"{title_prefix} - Mean Squared Error")
        axes[0].set_ylabel("Mean Squared Error")
        axes[0].tick_params(axis='x', rotation=45)

        sea.barplot(x="Model", y="R2", data=pd.DataFrame(results), ax=axes[1])
        axes[1].set_title(f"{title_prefix} - R2 Score")
        axes[1].set_ylabel("R2 Score")
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    plot_model_performance(results_rfe, "RFE Selected Features")
    plot_model_performance(results_pca, "PCA Selected Features")

data_manipulate(df)