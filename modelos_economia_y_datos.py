#Importación de librerías necesarias
import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

#Generación

conn = sqlite3.connect('laptop_prices_final.sqlite')
query = """
        SELECT Price_euros, TypeName, Inches, ResolutionNumber, GPU_brand, RAM_GB,
        HDD_space, SSD_space, Weight_KG FROM laptops_acotada
        WHERE CPU_brand != 'Samsung';
        """
df = pd.read_sql_query(query, conn)
conn.close()

df = pd.get_dummies(df, columns=['TypeName', 'GPU_brand'], drop_first=False)

X = df.drop(columns=['Price_euros'])  # Reemplaza 'target_column' por el nombre de tu columna objetivo
y = df['Price_euros']  # Si tienes una columna objetivo



#Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Estandarizar los datos
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

# Combinar ambas partes
X_train = np.hstack([X_train_scaled, X_train_dummies])
X_test = np.hstack([X_test_scaled, X_test_dummies])


#Definición de los parámetros para GridSearchCV

lasso_param_grid = {
    'alpha': [0.1, 0.25, 0.75, 0.5, 1, 2]  # Valores de alpha para probar
}

#GridSearchCV para KNN
knn = KNeighborsRegressor(n_neighbors=5)
#GridSearchCV para Random Forest
rf = RandomForestRegressor(random_state=42, n_estimators=100)
#GridSearchCV para Lasso
lasso_grid_search = GridSearchCV(Lasso(), lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

r2_scores = []
mse_scores = []

#KNN:
#Entrenar el modelo KNN
knn.fit(X_train, y_train)
#Hacer predicciones KNN
y_pred_knn = knn.predict(X_test)
#RESULTADOS KNN
mse = mean_squared_error(y_test, y_pred_knn)
r2 = r2_score(y_test, y_pred_knn)
print("Raiza de error cuadrático medio (RMSE):", mse**(0.5))
print("R^2 Score:", r2)
r2_scores.append(r2)
mse_scores.append(mse**(0.5))

print()

#Random Forest:
#Entrenar el modelo RF
rf.fit(X_train, y_train)
#Hacer predicciones RF
y_pred_rf = rf.predict(X_test)
#RESULTADOS rf
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
print("Raiza de error cuadrático medio (RMSE):", mse**(0.5))
print("R^2 Score:", r2)
r2_scores.append(r2)
mse_scores.append(mse**(0.5))


    
print()

#Lasso:
#Entrenar el modelo Lasso
lasso_grid_search.fit(X_train, y_train)
#Hacer predicciones Lasso
y_pred_lasso = lasso_grid_search.predict(X_test)
#RESULTADOS Lasso
mse = mean_squared_error(y_test, y_pred_lasso)
r2 = r2_score(y_test, y_pred_lasso)
print("Raiza de error cuadrático medio (RMSE):", mse**(0.5))
print("R^2 Score:", r2)

r2_scores.append(r2)
mse_scores.append(mse**(0.5))


#Imprimir los mejores hiperparámetros para Lasso
print("Mejores parámetros Lasso:")
print(lasso_grid_search.best_params_)
lasso_best_model = lasso_grid_search.best_estimator_
print("\nCoeficientes del modelo Lasso:")
print(pd.Series(lasso_best_model.coef_, index=df.drop(columns=['Price_euros']).columns))


import matplotlib.pyplot as plt
import seaborn as sns

# Crear el gráfico comparativo
fig, ax = plt.subplots(figsize=(10, 8))

# Lista de modelos y sus predicciones
model_predictions = {
    'KNN': y_pred_knn,
    'Random Forest': y_pred_rf,
    'Lasso': y_pred_lasso
}

# Graficar cada modelo
for name, y_pred in model_predictions.items():
    sns.scatterplot(x=y_test, y=y_pred, label=f"{name} (R²: {r2_score(y_test, y_pred):.2f})", ax=ax)

# Línea de referencia ideal
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal')

# Configuración del gráfico
ax.set_title("Comparación de valores reales vs predicciones")
ax.set_xlabel("Valores reales")
ax.set_ylabel("Predicciones")
ax.legend()
plt.show()

feature_names = X.columns
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nImportancia de las características RF:")
print(importance_df)

#Visualizar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las Características en Random Forest')
plt.gca().invert_yaxis()  # Invertir el eje para mostrar las más importantes primero
plt.show()

# Datos de los modelos
modelos = ['KNN', 'Random Forest', 'LASSO']


# Crear gráficos circulares para R^2 de cada modelo
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, (modelo, r2) in enumerate(zip(modelos, r2_scores)):
    axs[i].pie([r2, 1 - r2], labels=['R²', '1 - R²'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107'])
    axs[i].set_title(f'{modelo} (R² = {r2:.2f})')

# Ajustar espacio entre gráficos
plt.tight_layout()
plt.show()

# Crear gráfico de barras para MSE
plt.figure(figsize=(8, 6))
plt.bar(modelos, mse_scores, color=['#2196F3', '#4CAF50', '#FFC107'])
plt.title('RMSE de cada modelo')
plt.ylabel('RMSE')
plt.xlabel('Modelos')
plt.ylim(0, max(mse_scores) + 0.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
