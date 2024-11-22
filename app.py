# Para app.py
import streamlit as st
import pandas as pd
from predicciones import *
from carga_datos import procesar_datos

# Para gráficos

# Para exploratorio

#

tabs = st.tabs(["Inicio", "Prediccion", "Rendimiento de los modelos", "Analisis Exploratorio"])

with tabs[0]:
    st.header("Bienvenido al proyecto de economía y ciencia de datos del grupo 15")
    st.write("Análisis de precio de laptops")

with tabs[1]:
    st.header("Predicciones")
    # Cargar y procesar los datos
    # procesar_datos()  ->  Agrega y modifica columnas a partir del csv original
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Entrenar los modelos
    knn, rf, lasso_grid_search = train_models(X_train, y_train)
    
    # Interfaz de usuario con Streamlit
    st.title("Predicción de Precio de Computadores")
    st.write("Ingrese las características del computador para predecir su precio.")
    
    # Entradas del usuario
    # 1. Tipo de laptop
    tipo_laptop = st.radio(
        "Selecciona un tipo de laptop:",
        options=["Notebook", "Workstation", "Gaming", "Ultrabook", "2 in 1 Convertible", "Netbook"],
        index=0  # Opción seleccionada por defecto
    )
    
    # 2. Tamaño de la pantalla (pulgadas)
    inches = st.number_input(
        "Tamaño de la pantalla (Pulgadas):",
        min_value=10.0, max_value=20.0, value=15.6, step=0.1
    )
    
    # 3. Resolución en número (píxeles totales)
    resolution_number = st.number_input(
        "Número de píxeles en la resolución (ej, 1920x1080 -> 2073600):",
        min_value=0, value=2073600
    )
    
    # 4. Marca de GPU
    gpu_brand = st.radio(
        "Selecciona la marca del GPU:",
        options=["NVIDIA", "AMD", "Intel", "Otro"],
        index=0
    )
    
    # 5. Marca de CPU
    cpu_brand = st.radio(
        "Selecciona la marca del CPU:",
        options=["Intel", "AMD", "Otro"],
        index=0
    )
    
    # 6. RAM (GB)
    ram_gb = st.number_input(
        "Cantidad de RAM (GB):",
        min_value=1, max_value=128, value=8
    )
    
    # 7. Espacio en disco duro (HDD) en GB
    hdd_space = st.number_input(
        "Espacio en HDD (GB, dejar como 0 si no tiene):",
        min_value=0, max_value=2048, value=0, step=1
    )
    
    # 8. Espacio en disco sólido (SSD) en GB
    ssd_space = st.number_input(
        "Espacio en SSD (GB, dejar como 0 si no tiene):",
        min_value=0, max_value=2048, value=256, step=1
    )
    
    # 9. Peso en KG
    weight_kg = st.number_input(
        "Peso del computador (KG):",
        min_value=0.5, max_value=10.0, value=1.5, step=0.1
    )
    
    # Botón para procesar
    if st.button("Procesar"):
        if hdd_space == 0 and ssd_space == 0:
            st.error("HDD y SSD no pueden ser 0 al mismo tiempo.")
        else:
            # Mostrar los datos ingresados
            st.write("### Datos ingresados:")
            st.write(f"- Tipo de Laptop: {tipo_laptop}")
            st.write(f"- Tamaño de la pantalla (Pulgadas): {inches}")
            st.write(f"- Resolución (Número de píxeles): {resolution_number}")
            st.write(f"- Marca del GPU: {gpu_brand}")
            st.write(f"- Marca del CPU: {cpu_brand}")
            st.write(f"- RAM (GB): {ram_gb}")
            st.write(f"- Espacio en HDD (GB): {hdd_space}")
            st.write(f"- Espacio en SSD (GB): {ssd_space}")
            st.write(f"- Peso (KG): {weight_kg}")
    
            # Predicción
            input_data = pd.DataFrame({
                'Inches': [inches],
                'ResolutionNumber': [resolution_number],
                'RAM_GB': [ram_gb],
                'HDD_space': [hdd_space],
                'SSD_space': [ssd_space],
                'Weight_KG': [weight_kg],
                'TypeName_Notebook': [1 if tipo_laptop == 'Notebook' else 0],
                'TypeName_Netbook': [1 if tipo_laptop == 'Netbook' else 0],
                'TypeName_Workstation': [1 if tipo_laptop == 'Workstation' else 0],
                'TypeName_Gaming': [1 if tipo_laptop == 'Gaming' else 0],
                'TypeName_2 in 1 Convertible': [1 if tipo_laptop == '2 in 1 Convertible' else 0],
                'TypeName_Ultrabook': [1 if tipo_laptop == 'Ultrabook' else 0],
                'GPU_brand_Nvidia': [1 if gpu_brand == 'NVIDIA' else 0],
                'GPU_brand_AMD': [1 if gpu_brand == 'AMD' else 0],
                'GPU_brand_Intel': [1 if gpu_brand == 'Intel' else 0]
            })
    
            price_knn, price_rf, price_lasso = predict_price(
                (knn, rf, lasso_grid_search), input_data, scaler, ['Inches', 'ResolutionNumber', 'RAM_GB', 'HDD_space', 'SSD_space', 'Weight_KG'],
                ['TypeName_Notebook', 'TypeName_Netbook', 'TypeName_Workstation', 'TypeName_Gaming', 'TypeName_2 in 1 Convertible', 'TypeName_Ultrabook', 'GPU_brand_Nvidia', 'GPU_brand_AMD', 'GPU_brand_Intel']
            )
    
            st.write("### Predicciones del precio del computador:")
            st.write(f"Predicción con KNN: {price_knn:.2f} EUR")
            st.write(f"Predicción con Random Forest: {price_rf:.2f} EUR")
            st.write(f"Predicción con Lasso: {price_lasso:.2f} EUR")
            
with tabs[2]:
    st.header("Rendimiento de los modelos")
    # Cargar los datos y procesarlos
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Entrenar los modelos
    knn, rf, lasso_cv = train_models(X_train, y_train)
    
    # Predicciones para los tres modelos
    y_pred_knn = knn.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_lasso = lasso_cv.predict(X_test)
    
    # Generar los gráficos
    fig1, importance_fig, fig_lasso_cv, fig_r2_knn, fig_r2_rf, fig_r2_lasso, fig_rmse = plot_model_performance(df, y_test, y_pred_knn, y_pred_rf, y_pred_lasso, rf, lasso_cv)
    
    # Mostrar los gráficos en Streamlit
    st.plotly_chart(fig1)
    st.plotly_chart(importance_fig)
    st.plotly_chart(fig_lasso_cv)
    st.plotly_chart(fig_r2_knn)
    st.plotly_chart(fig_r2_rf)
    st.plotly_chart(fig_r2_lasso)
    st.plotly_chart(fig_rmse)


with tabs[3]:
    st.header("Analisis Exploratorio")
