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
df = pd.read_csv("./winequality-red.csv")

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
    X = df.drop("quality", axis=1)  # Özellikleri ayırıyoruz
    y = df["quality"]  # Hedef değişkeni ayırıyoruz

    # Eğitim ve test setlerine ayırıyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Veriyi ölçeklendirmek için fonksiyon
    def scale_data(X_train, X_test, scaler):
        scaler.fit(X_train)  # Eğitim setine göre ölçeklendiriciyi ayarlıyoruz
        X_train_scaled = scaler.transform(X_train)  # Eğitim setini ölçeklendiriyoruz
        X_test_scaled = scaler.transform(X_test)  # Test setini ölçeklendiriyoruz
        return X_train_scaled, X_test_scaled

    # MinMaxScaler ile ölçeklendirme
    X_train_scaled_norm, X_test_scaled_norm = scale_data(X_train, X_test, MinMaxScaler())
    # StandardScaler ile ölçeklendirme
    X_train_scaled_std, X_test_scaled_std = scale_data(X_train, X_test, StandardScaler())

    # RFE ile özellik seçimi
    def rfe_feature_selection(X_train, y_train):
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=5)  # 5 özellik seçiyoruz
        rfe.fit(X_train, y_train)
        return rfe.transform(X_train), rfe.transform(X_test)

    # PCA ile özellik seçimi
    def pca_feature_selection(X_train, X_test):
        pca = PCA(n_components=5)  # 5 bileşen seçiyoruz
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca

    X_train_rfe, X_test_rfe = rfe_feature_selection(X_train_scaled_norm, y_train)
    X_train_pca, X_test_pca = pca_feature_selection(X_train_scaled_norm, X_test_scaled_norm)

    # Parametre grid'leri tanımlıyoruz
    param_grids = {
        "Linear Regression": {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "positive": [True, False]
        },
        "Decision Tree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 5]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        },
        "Support Vector Machine": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"]
        },
        "Neural Network": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "learning_rate": ["constant", "adaptive"],
            "max_iter": [200, 500, 1000]
        }
    }

    # Modelleri tanımlıyoruz
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=0),
        "Support Vector Machine": SVR(),
        "Neural Network": MLPRegressor(random_state=0)
    }

    # Grid search ile model seçim fonksiyonu
    def perform_grid_search(model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    # Random search ile model seçim fonksiyonu
    def perform_random_search(model, param_grid, X_train, y_train):
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', random_state=0)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

    # Sonuçları saklayacağımız sözlükler
    results_rfe = {"Model": [], "MSE": [], "R2": []}
    results_pca = {"Model": [], "MSE": [], "R2": []}

    # RFE özellikleri için model performansını değerlendiriyoruz
    for name, model in models.items():
        best_model = perform_grid_search(model, param_grids[name], X_train_rfe, y_train)
        y_pred_rfe = best_model.predict(X_test_rfe)
        results_rfe["Model"].append(name)
        results_rfe["MSE"].append(mean_squared_error(y_test, y_pred_rfe))
        results_rfe["R2"].append(r2_score(y_test, y_pred_rfe))

    # PCA özellikleri için model performansını değerlendiriyoruz
    for name, model in models.items():
        best_model = perform_random_search(model, param_grids[name], X_train_pca, y_train)
        y_pred_pca = best_model.predict(X_test_pca)
        results_pca["Model"].append(name)
        results_pca["MSE"].append(mean_squared_error(y_test, y_pred_pca))
        results_pca["R2"].append(r2_score(y_test, y_pred_pca))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Comparison')

# RFE seçilmiş özellikler için model performansını karşılaştıran grafikleri çiziyoruz
    def plot_model_performance(results, title_prefix):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title_prefix} Model Performans Karşılaştırması')

        sea.barplot(x="Model", y="MSE", data=pd.DataFrame(results), ax=axes[0])
        axes[0].set_title(f"{title_prefix} - Ortalama Kare Hatası")
        axes[0].set_ylabel("Ortalama Kare Hatası")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

        sea.barplot(x="Model", y="R2", data=pd.DataFrame(results), ax=axes[1])
        axes[1].set_title(f"{title_prefix} - R2 Skoru")
        axes[1].set_ylabel("R2 Skoru")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    # RFE özellikleri ile performansı karşılaştırıyoruz
    plot_model_performance(results_rfe, "RFE Seçilmiş Özellikler")

    # PCA özellikleri ile performansı karşılaştırıyoruz
    plot_model_performance(results_pca, "PCA Seçilmiş Özellikler")


data_manipulate(df)