
###### Librerias

import pandas as pd
import numpy as np
import time
import matplotlib
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
# pip install xlrd para leer excel
# pip install scikit-learn para el modelo de predicción
# pip install sklearn
# pip install lightgbm
# pip install skforecast
# pip install statsmodels
# pip install jinja2
######  Exploración de datos

df = pd.read_excel('Sample_Superstore.xls')
#a=df.describe(include='all')
#print(a)
#c=df.describe(include=[object])
#print(c)
# Hay 21 columnas
#b=df.columns.values
# Visualizar 
#for i in b:
#    print(i)
#    print(df[i])
print(df.describe(include='all'))
print(df.shape)
print(df.isnull().sum())
print(df.isna().sum())
df = df.drop_duplicates()
print(df.shape)
# No hay datos faltantes ni duplicados

########## Agregando La columna Year, mes y dia
df['Year']=[i.year for i in df['Order Date']]
df['Month']=[i.month for i in df['Order Date']]
df['Name_month']=[datetime.strptime(str(i),'%m').strftime('%B') for i in df['Month']]
days=['Mon','Tues','Wednes','Thurs','Fri','Satur','Sun']
df['Day']=[days[i.weekday()]+"day" for i in df['Order Date']]

######### Ordenando por años
df1=df.sort_values(by=['Year'])
d_year=df1['Year'].unique()

#plotdata = pd.DataFrame(
#    {"Sales": Sales_total}, 
#    {"Profit": Sales_total}, 
#    index=c1)
#df.plot(kind='bar')
########## Ordenando por meses
df2=df.sort_values(by=['Month'])
d_month=df2['Month'].unique()
d_month=[datetime.strptime(str(i),'%m').strftime('%B') for i in d_month]
# strptime- convierte a formato de fecha '%d/%m/%Y' 
# strftime- convierte a nombres los meses "%d %B, %Y"


########## Ordenando por Dias
df3=df.sort_values(by=['Day'])
d_day=df3['Day'].unique()
days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
########## función para Plotear

def plot_easy(c,df,df1,world):
    m=str(world)
    c1=[]
    Sales_total=[]
    Profit_total=[]
    for i in c:
        df1=df[df[m]==i]
        Sales_total.append(df1['Sales'].sum())
        Profit_total.append(df1['Profit'].sum())
        c1.append(str(i))  
    df11 = pd.DataFrame()
    df11[m]=c1 
    df11['Sales']=Sales_total 
    df11['Profit']=Profit_total 
    fig,axs=plt.subplots(3,1, figsize=[15,20])
    axs[2].bar(df11[m],df11['Sales'] ,label = "Sales")
    axs[2].bar(df11[m],df11['Profit'] , label = "Profit")
    axs[2].set_xlabel(m)
    axs[2].set_ylabel("Sales Vs Profit")
    axs[2].legend()
    axs[1].bar(df11[m],df11['Sales'] ,label = "Sales")
    #axs[1].set_xlabel("Años")
    axs[1].set_ylabel("Sales")
    axs[1].legend()
    axs[0].bar(df11[m],df11['Profit'] , label = "Profit")
    #axs[0].set_xlabel("Años")
    axs[0].set_ylabel("Profit")
    axs[0].legend()
    axs[0].set_title('Sales Vs Profit')
    plt.show()

#plot_easy(d_year,df,df1,'Year')
#plot_easy(d_month,df,df2,'Name_month')
#plot_easy(days,df,df2,'Day')

#######
df4=df.sort_values(by=['Order Date'])
plt.figure(figsize=[15,20])
plt.plot(np.linspace(0,df4.shape[0],df4.shape[0]),df4['Sales'], 'bo')
#plt.show()


###### Investigando valor con Sales mas grande
max_graphic = df[df['Sales']==df['Sales'].max()]['Order Date'].to_frame()
max_graphic.reset_index(drop=True, inplace=True)
print(df['Sales'].max()) 
df_sales_max = df[df['Order Date']==max_graphic['Order Date'][0]]  
print(df_sales_max[['Order Date','Sales','Quantity','Profit','Product ID','Product Name','Discount']])

####### Explorando el producto para buscar inconsistencias
product = df[df['Sales']==df['Sales'].max()]['Product ID'].to_frame()
print(product)
product.reset_index(drop=True, inplace=True)
df_product = df[df['Product ID']==product['Product ID'][0]]  
print(df_product)


####### Lista de valores sin profit negativo
df5=df.sort_values(by=['Order Date'])
df5=df5[df5['Profit']>=0]
plt.figure(figsize=[15,20])
plt.plot(np.linspace(0,df5.shape[0],df5.shape[0]),df5['Sales'], 'bo')
plt.show()

######## Datos acumulados por dias
df6=df.groupby(by='Order Date').sum()
print(df6)
plt.figure(figsize=[15,20])
plt.plot(np.linspace(0,df6.shape[0],df6.shape[0]),df6['Sales'], 'bo')
#plt.show()

###### acumulados por semanas del año
df7=df
df7['week']=[i.isocalendar().week for i in df['Order Date']]
df7=df7.groupby(by='week').sum()
plt.figure(figsize=[15,20])
plt.plot(np.linspace(0,df7.shape[0],df7.shape[0]),df7['Sales'], 'bo')
#plt.show()


####### Analisis por productos 1862 productos y 1850 nombres
d_product=df['Product ID'].unique()
print(d_product)
name_product=df['Product Name'].unique()
print(name_product)
df8=df[['Product Name','Product ID','Sales','Quantity','Profit']].groupby(by='Product Name').sum()
#df8=df8.sort_values(by=['Quantity'],ascending=False)
print(df8)
df8 = df8[0:5]
#print(df8['Product Name'])
# fig,axs=plt.subplots(3,1, figsize=[15,20])
# axs[2].bar(df8['Sales'] ,label = "Sales")
# axs[2].bar(df8['Profit'] , label = "Profit")
# axs[2].set_ylabel("Sales Vs Profit")
# axs[2].legend()
# axs[1].bar(df8['Sales'] ,label = "Sales")
# #axs[1].set_xlabel("Años")
# axs[1].set_ylabel("Sales")
# axs[1].legend()
# axs[0].bar(df8['Profit'] ,label = "Profit")
# #axs[0].set_xlabel("Años")
# axs[0].set_ylabel("Profit")
# axs[0].legend()
# axs[0].set_title('Sales Vs Profit')
# plt.show()

df_serie_temp=df[['Order Date','Year','Month','Quantity','Discount','Profit','Sales']]
df_serie_temp=df_serie_temp.sort_values(by=['Order Date'])
df_serie_temp.reset_index(inplace=True, drop=True)
df_serie_temp=df_serie_temp.groupby(['Order Date']).mean().reset_index()
print(df_serie_temp)
print(365*4)

year = np.array([2014,2015,2016,2017],float)
month = np.linspace(1,12,12)
print(month)
for i in year:
    for j in month:
        df_serie_temp_14 = df_serie_temp[df_serie_temp['Year']==i]
        df_serie_temp_14 = df_serie_temp_14[df_serie_temp_14['Month']==j]
        print(i,j)
        print(df_serie_temp_14.shape[0])


from sklearn.model_selection import train_test_split

# Split the dataset in an 75/25 train/test ratio. 
train, test = train_test_split(df_serie_temp, test_size=0.25, random_state=10)

print(train['Order Date'])
print("Train size:", train.shape[0])
print("Test size:", test.shape[0])

###

