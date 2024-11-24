#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:09:40 2024

@author: diegomanriquez
"""

import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Conectar a la base de datos SQLite
conn = sqlite3.connect('laptop_prices.sqlite')

# Usar read_sql para leer los datos
df = pd.read_sql("SELECT * FROM laptops", conn)

# Cerrar la conexión
conn.close()

# Mostrar el DataFrame
print(df)

##Head del DF
head= df.head(10)

##describe del precio##

df['Price_euros'].describe()

##Histograma del precio## codigo obtenido de link: https://www.kaggle.com/code/abonaplata/analisis-exploratorio-de-datos-con-python?scriptVersionId=17204829&cellId=21
sns.distplot(df['Price_euros'])

##Matriz de corr## codigo obtenido de link: https://www.kaggle.com/code/abonaplata/analisis-exploratorio-de-datos-con-python?scriptVersionId=17204829&cellId=21
df_numerico = df.select_dtypes(include=['number'])
corrmat = df_numerico.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


##Grafico de cajas## codigo obtenido de link: https://www.kaggle.com/code/abonaplata/analisis-exploratorio-de-datos-con-python?scriptVersionId=17204829&cellId=21
var = 'Hybrid_space'
data = pd.concat([df['Price_euros'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x=var, y="Price_euros", data=data)
fig.axis(ymin=0, ymax=7000)


##Grafico de dispersion## codigo obtenido de link: https://www.kaggle.com/code/mohamedhaithamyamani/laptop-price-prediction-with-eda?scriptVersionId=197820240&cellId=14
plt.figure(figsize=(7,10)) 
sns.scatterplot(df, x='Price_euros', y='ScreenResolution')
plt.show()











