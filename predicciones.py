# predictor.py
import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


# Conexión a la base de datos SQLite y consulta
def load_data():
    conn = sqlite3.connect('laptop_prices_final.sqlite')
    query = """
            SELECT Price_euros, TypeName, Inches, ResolutionNumber, GPU_brand, RAM_GB,
            HDD_space, SSD_space, Weight_KG FROM laptops_acotada
            WHERE CPU_brand != 'Samsung';
            """
    df = pd.read_sql_query(query, conn)
    df = pd.get_dummies(df, columns=['TypeName', 'GPU_brand'], drop_first=False)
    conn.close()
    return df


# Preprocesamiento de los datos
def preprocess_data(df):
    X = df.drop(columns=['Price_euros'])
    y = df['Price_euros']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar las columnas numéricas
    numeric_columns = ['Inches', 'ResolutionNumber', 'RAM_GB', 'HDD_space', 'SSD_space', 'Weight_KG']
    dummy_columns = [col for col in X.columns if col not in numeric_columns]

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
    lasso_cv = LassoCV(cv=5)

    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lasso_cv.fit(X_train, y_train)

    return knn, rf, lasso_cv


# Realizar predicción con los modelos entrenados
def predict_price(models, input_data, scaler, numeric_columns, dummy_columns):
    knn, rf, lasso_cv = models

    # Estandarizar las características numéricas
    input_data_scaled = scaler.transform(input_data[numeric_columns])
    input_data_dummies = input_data[dummy_columns].values
    input_data_final = np.hstack([input_data_scaled, input_data_dummies])

    # Predicciones
    price_knn = knn.predict(input_data_final)[0]
    price_rf = rf.predict(input_data_final)[0]
    price_lasso = lasso_cv.predict(input_data_final)[0]

    return price_knn, price_rf, price_lasso


# Generar gráficos de desempeño de los modelos
def plot_model_performance(df, y_test, y_pred_knn, y_pred_rf, y_pred_lasso, rf_model, lasso_cv):

    #Genera gráficos interactivos comparando los valores reales con las predicciones
    #y visualiza la importancia de las características y el desempeño de los modelos.
    
    # 1. Comparación de valores reales vs. predicciones (scatterplot)
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=y_test, y=y_pred_knn, mode='markers', name='Predicción KNN', marker=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=y_test, y=y_pred_rf, mode='markers', name='Predicción Random Forest', marker=dict(color='green')))
    fig1.add_trace(go.Scatter(x=y_test, y=y_pred_lasso, mode='markers', name='Predicción Lasso', marker=dict(color='red')))
    fig1.add_trace(go.Scatter(
        x=y_test, 
        y=y_test,  # Línea y = x
        mode='lines', 
        name='Predicción = valor real', 
        line=dict(color='yellow', dash='dash')
    ))
    fig1.update_layout(
        title="Comparación entre valores reales y predicciones",
        xaxis_title="Valores reales",
        yaxis_title="Predicciones",
        showlegend=True
    )

    # 2. Importancia de las características (Random Forest)
    feature_importances = rf_model.feature_importances_
    feature_names = df.drop(columns=['Price_euros']).columns
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_feature_importances = feature_importances[sorted_idx]
    sorted_feature_names = feature_names[sorted_idx]

    importance_fig = go.Figure(go.Bar(
        y=sorted_feature_names[::-1], 
        x=sorted_feature_importances[::-1], 
        orientation='h', 
        marker=dict(color='rgb(50, 50, 250)')
    ))
    importance_fig.update_layout(
        title="Importancia de las características de Random Forest",
        xaxis_title="Importancia",
        yaxis_title="Características"
    )

    # 3. Visualización de regularización Lasso
    N_FOLD = 5
    mean_alphas = lasso_cv.mse_path_.mean(axis=-1)

    fig_lasso_cv = go.Figure([go.Scatter(
        x=lasso_cv.alphas_, y=lasso_cv.mse_path_[:, i],
        name=f"Fold: {i+1}", opacity=.5, line=dict(dash='dash'),
        hovertemplate="alpha: %{x} <br>MSE: %{y}"
    ) for i in range(N_FOLD)])

    fig_lasso_cv.add_traces(go.Scatter(
        x=lasso_cv.alphas_, y=mean_alphas,
        name='Mean', line=dict(color='yellow', width=1),
        hovertemplate="alpha: %{x} <br>MSE: %{y}",
    ))

    fig_lasso_cv.add_shape(
        type="line", line=dict(dash='dash', color='white'),
        x0=lasso_cv.alpha_, y0=0,
        x1=lasso_cv.alpha_, y1=1,
        yref='paper'
    )

    fig_lasso_cv.update_layout(
        xaxis=dict(
            title=dict(text='alpha'),
            type='log'
        ),
        yaxis=dict(title=dict(text='Mean Square Error (MSE)'))
    )
    fig_lasso_cv.update_layout(title="Visualización de regularización Lasso")

    # 4. Gráficos circulares de R² y RMSE para cada modelo

    r2_knn = r2_score(y_test, y_pred_knn)
    r2_rf = r2_score(y_test, y_pred_rf)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    # R2 knn
    fig_r2_knn = go.Figure(data=[go.Pie(
        labels=["R²", "1-R²"], values=[r2_knn, 1 - r2_knn], hole=0.3
    )])
    fig_r2_knn.update_layout(title="R² de KNN")
    # R2 rf
    fig_r2_rf = go.Figure(data=[go.Pie(
        labels=["R²", "1-R²"], values=[r2_rf, 1 - r2_rf], hole=0.3
    )])
    fig_r2_rf.update_layout(title="R² de Random Forest")
    # R2 lasso
    fig_r2_lasso = go.Figure(data=[go.Pie(
        labels=["R²", "1-R²"], values=[r2_lasso, 1 - r2_lasso], hole=0.3
    )])
    fig_r2_lasso.update_layout(title="R² de Lasso")

    # RMSE's
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

    fig_rmse = go.Figure(data=[go.Bar(
        x=["KNN", "Random Forest", "Lasso"],
        y=[rmse_knn, rmse_rf, rmse_lasso],
        marker=dict(color=['blue', 'green', 'red'])
    )])
    fig_rmse.update_layout(
        title="RMSE de los modelos",
        xaxis_title="Modelos",
        yaxis_title="RMSE"
    )

    # Retornar los gráficos
    return fig1, importance_fig, fig_lasso_cv, fig_r2_knn, fig_r2_rf, fig_r2_lasso, fig_rmse



