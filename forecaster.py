import pandas as pd
import numpy as np
import time
import matplotlib
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
import warnings
warnings.filterwarnings('once')

def plot_easy_v2(c,df,df1,world,world1,world2):
    m=str(world)
    m1=str(world1)
    m2=str(world2)
    c1=[]
    Sales_total=[]
    Profit_total=[]
    for i in c:
        df1=df[df[m]==i]
        Sales_total.append(df1[m1].sum())
        Profit_total.append(df1[m2].sum())
        c1.append(str(i))  
    df11 = pd.DataFrame()
    df11[m]=c1 
    df11[m1]=Sales_total 
    df11[m2]=Profit_total 
    plt.figure(figsize=[5,5])
    plt.bar(df11[m],df11[m1] ,0.4,label = m1)
    plt.bar(df11[m],df11[m2] , 0.4 ,label = m2)
    plt.xticks(rotation=45)
    plt.xlabel(m)
    m3=str(m1+' vs '+m2)
    plt.ylabel(m3)
    plt.legend()
    plt.title(m3)
    plt.show()
    
df = pd.read_excel('Sample_Superstore.xls')
table_name = df.columns.values
print(table_name)

null = df.isnull().sum() #suma las entradas nulas por cada variable
print('Cantidad de datos vacios:', null.sum()) # imprime el total
is_nan=df.isna().sum() #suma las entradas NaN por cada variable
print('Cantidad de datos NaN:',is_nan.sum()) # imprime el total
df = df.drop_duplicates() # elimina duplicados
print('Cantidad de filas: {} Cantidad de variables: {}'.format(df.shape[0],df.shape[1]))

df_serie_temp=df[['Order Date','Quantity','Discount','Profit','Sales']]
df_serie_temp=df_serie_temp.sort_values(by=['Order Date'])
df_serie_temp.reset_index(inplace=True, drop=True)
df_serie_temp=df_serie_temp.groupby(['Order Date']).mean().reset_index()
df_serie_temp_1 = df_serie_temp.set_index('Order Date').sort_index().copy()
df_serie_temp_1 = df_serie_temp_1.asfreq(freq='D')
print(f"Filas con missing values: {df_serie_temp_1.isnull().any(axis=1).sum()}")
df_serie_temp_1 = df_serie_temp_1.fillna(method='ffill')
print(df_serie_temp_1.head(10))
print(df_serie_temp_1.shape)

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
fig, ax = plt.subplots(figsize=(5, 2))
plot_acf(df_serie_temp_1.Sales, ax=ax, lags=40)
plt.show()

fig, ax = plt.subplots(figsize=(5, 2))
plot_pacf(df_serie_temp_1.Profit, ax=ax, lags=60)
plt.show()

total_1 = int(df_serie_temp_1.shape[0]*0.7)
total_2 = int(df_serie_temp_1.shape[0]*0.2)
total_3 = int(df_serie_temp_1.shape[0]*0.1)
if (total_1+total_2+total_3)-df_serie_temp_1.shape[0]!=0:
    total_1=total_1+np.abs(total_1+total_2+total_3-df_serie_temp_1.shape[0])
fin_train = df_serie_temp_1.index[total_1-1]
fin_validacion = df_serie_temp_1.index[(total_1+total_2)-1]
print(fin_train)
print(fin_validacion)

datos_train = df_serie_temp_1.loc[ :fin_train, : ].copy()
datos_val   = df_serie_temp_1.loc[fin_train:fin_validacion, :].copy()
datos_test  = df_serie_temp_1.loc[fin_validacion:, :].copy()

fig, ax = plt.subplots(figsize=(8, 3.5))
datos_train.Profit.plot(ax=ax, label='Entrenamiento', linewidth=1)
datos_val.Profit.plot(ax=ax, label='Validación', linewidth=1)
datos_test.Profit.plot(ax=ax, label='Test', linewidth=1)
plt.xticks(rotation=40)
ax.set_title('Sales')
ax.legend()

forecaster = ForecasterAutoreg(
                 regressor     = Ridge(),
                 lags          = 30,
                 transformer_y = StandardScaler()
             )

forecaster.fit(y=df_serie_temp_1.loc[:fin_validacion, 'Profit'])
forecaster

metrica, predicciones = backtesting_forecaster(
                            forecaster         = forecaster,
                            y                  = df_serie_temp_1['Profit'],
                            steps              = 30,
                            metric             = 'mean_absolute_error',
                            initial_train_size = len(df_serie_temp_1.loc[:fin_validacion]),
                            refit              = False,
                            verbose            = True,
                            show_progress      = True
                        )

fig, ax = plt.subplots(figsize=(8, 3.5))
df_serie_temp_1.loc[predicciones.index, 'Profit'].plot(ax=ax, linewidth=2, label='test')
predicciones.plot(linewidth=2, label='predicción', ax=ax)
ax.set_title('Predicción vs demanda real')
ax.legend()

print(f'Error backtest: {metrica}')

# Normalización
# for column in df_serie_temp.columns:
#     if column != 'Order Date':
#         df_serie_temp[column] = (df_serie_temp[column] - df_serie_temp[column].min())/(df_serie_temp[column].max() - df_serie_temp[column].min())
#         df_serie_temp[column] = df_serie_temp[column]  / df_serie_temp[column].abs().max()
#         df_serie_temp[column] = (df_serie_temp[column] - df_serie_temp[column].mean()) / df_serie_temp[column].std()
# print(df_serie_temp)