# predictor.py
import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Conexión a la base de datos SQLite y consulta
def load_data():
    conn = sqlite3.connect('laptop_prices_final.sqlite')
    query = """
            SELECT Price_euros, TypeName, Inches, ResolutionNumber, GPU_brand, RAM_GB,
            HDD_space, SSD_space, Weight_KG FROM laptops_acotada
            WHERE CPU_brand != 'Samsung';
            """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Preprocesamiento de los datos
def preprocess_data(df):
    df = pd.get_dummies(df, columns=['TypeName', 'GPU_brand'], drop_first=False)

    X = df.drop(columns=['Price_euros'])
    y = df['Price_euros']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar los datos
    # Seleccionar columnas
    numeric_columns = ['Inches', 'ResolutionNumber', 'RAM_GB', 'HDD_space', 'SSD_space', 'Weight_KG']
    dummy_columns = [col for col in X.columns if col not in numeric_columns]

    # Estandarizar columnas numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled = scaler.transform(X_test[numeric_columns])

    # Conservar columnas dummies
    X_train_dummies = X_train[dummy_columns].values
    X_test_dummies = X_test[dummy_columns].values

    # Combinar columnas numéricas estandarizadas con columnas dummies
    X_train = np.hstack([X_train_scaled, X_train_dummies])
    X_test = np.hstack([X_test_scaled, X_test_dummies])

    return X_train, X_test, y_train, y_test, scaler, X.columns

# Entrenamiento de los modelos
def train_models(X_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=5)
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    lasso_param_grid = {'alpha': [0.1, 0.25, 0.5, 1, 2]}
    lasso_grid_search = GridSearchCV(Lasso(), lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lasso_grid_search.fit(X_train, y_train)

    return knn, rf, lasso_grid_search

# Realizar predicción con los modelos entrenados
def predict_price(models, input_data, scaler, numeric_columns, dummy_columns):
    knn, rf, lasso_grid_search = models

    # Estandarizar las características numéricas
    input_data_scaled = scaler.transform(input_data[numeric_columns])
    input_data_dummies = input_data[dummy_columns].values
    input_data_final = np.hstack([input_data_scaled, input_data_dummies])

    # Predicciones
    price_knn = knn.predict(input_data_final)[0]
    price_rf = rf.predict(input_data_final)[0]
    price_lasso = lasso_grid_search.predict(input_data_final)[0]

    return price_knn, price_rf, price_lasso
