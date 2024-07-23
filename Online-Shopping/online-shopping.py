import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------Data Analysz-------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------------#
df=pd.read_csv("./Online-Shopping/file.csv")
df=df.dropna()
df["profit"]=df["Offline_Spend"]+df["Online_Spend"]

def veri_analizi(df):
    plt.figure(figsize=(15, 12))

    # 1. Cinsiyet Dağılımı
    plt.subplot(2, 2, 1)
    sex = df.groupby("Gender")["profit"].sum()
    sns.barplot(x=sex.index, y=sex.values, color="r")
    plt.title("Cinsiyetlere Göre Toplam Harcama")

    # 2. Lokasyon Analizi
    plt.subplot(2, 2, 2)
    location = df.groupby("Location")["profit"].sum().sort_values()
    sns.barplot(x=location.index, y=location.values, color="r")
    plt.title("Lokasyonlara Göre Toplam Harcama")
    plt.xticks(rotation=90)

    # 3. Tenure ve Harcamalar Arasındaki İlişki
    plt.subplot(2, 2, 3)
    sns.scatterplot(x="Tenure_Months", y="profit", data=df)
    plt.title("Tenure_Months ve Toplam Harcama Arasındaki İlişki")

    # 4. Kupon Kullanımı
    plt.subplot(2, 2, 4)
    coupon_status = df.groupby("Coupon_Status")["profit"].sum().sort_values()
    sns.barplot(x=coupon_status.index, y=coupon_status.values, color="r")
    plt.title("Kupon Kullanımına Göre Toplam Harcama")

    plt.tight_layout()
    plt.show()

#################################################################################################################################################
#----------------------------------------------------------------Linner Regression--------------------------------------------------------------#
#################################################################################################################################################
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor




def profit(a):
    a=a[['Gender', 'Location', 'Tenure_Months',"profit"]]
    a = pd.get_dummies(a, columns=['Gender', 'Location', 'Tenure_Months'])
    print(a)
    X=a.drop("profit",axis=1)
    y=a["profit"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    Model_Linner=LinearRegression()
    Model_Linner.fit(X_train,y_train)
    y_pred=Model_Linner.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(r2)

def fiyat_etkisi(a):
    x=["Avg_Price","Delivery_Charges","profit"]
    corr=a[x].corr()
    mask=np.triu(np.ones_like(corr,type(bool)))
    sns.heatmap(corr,mask=mask,fmt=".2f",annot=True)
    plt.show()

def indirim_etkisi(a):
    columns_of_interest = ["Coupon_Status", "Discount_pct", "profit"]

    a = a[columns_of_interest]
    a = pd.get_dummies(a, columns=["Coupon_Status"], drop_first=True)
    corr = a.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(corr, mask=mask, fmt=".2f", annot=True, cmap='coolwarm', center=0)
    plt.title('Korelasyon Matrisinin Heatmap Görselleştirmesi') 
    plt.show()

def aylik_tahmin(a):

    a['Date'] = pd.to_datetime(a['Date'])
    a['Month'] = a['Date'].dt.to_period('M')
    monthly_data = a.groupby('Month').agg({'profit': 'sum'}).reset_index()
    X = np.arange(len(monthly_data)).reshape(-1, 1)  # Ay sayısı
    y = monthly_data['profit'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Aylık Tahmin R^2 Skoru: {r2}")


def multicollinearity_check(a):
   
    X = a[['Avg_Price', 'Delivery_Charges', 'Discount_pct', 'Tenure_Months']]
    X = pd.get_dummies(X, drop_first=True)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print(vif_data)

"""veri_analizi(df)"""
profit(df)
"""fiyat_etkisi(df)"""
"""indirim_etkisi(df)
aylik_tahmin(df)
multicollinearity_check(df)"""