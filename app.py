import streamlit as st
import pandas as pd
from predicciones import *
from carga_datos import procesar_datos

#procesar_datos()

# Configuración de las pestañas
tabs = st.tabs(["Inicio", "Prediccion", "Rendimiento de los modelos", "Analisis Exploratorio"])

# Pestaña de inicio
with tabs[0]:
    st.header("¡Bienvenido al Proyecto de Economía y Ciencia de Datos del Grupo 15!")
    st.write("En este proyecto, realizamos un análisis sobre los precios de computadores, con un enfoque detallado en sus características 💻💲.")

# Pestaña de predicción
with tabs[1]:
    # Cargar y procesar los datos
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # Entrenar los modelos
    knn, rf, lasso_grid_search = train_models(X_train, y_train)

    # Interfaz de usuario con Streamlit
    st.title("Predicción de Precio de Computadores")
    st.write("Ingrese las características del computador para predecir su precio.")

    # Entradas del usuario
    tipo_laptop = st.radio("Selecciona un tipo de laptop:", options=["Notebook", "Workstation", "Gaming", "Ultrabook", "2 in 1 Convertible", "Netbook"], index=0)
    inches = st.number_input("Tamaño de la pantalla (Pulgadas):", min_value=9.0, max_value=21.0, value=15.6, step=0.1)
    ancho = st.number_input("Introduce el ancho de la pantalla (en píxeles):", min_value=100, max_value=10000, step=1)
    alto = st.number_input("Introduce el alto de la pantalla (en píxeles):", min_value=100, max_value=10000, step=1)
    resolution_number = ancho * alto
    gpu_brand = st.radio("Selecciona la marca del GPU:", options=["NVIDIA", "AMD", "Intel", "Otro"], index=0)
    cpu_brand = st.radio("Selecciona la marca del CPU:", options=["Intel", "AMD", "Otro"], index=0)
    ram_gb = st.number_input("Cantidad de RAM (GB):", min_value=1, max_value=128, value=8)
    hdd_space = st.number_input("Espacio en HDD (GB, dejar como 0 si no tiene):", min_value=0, max_value=2048, value=0, step=1)
    ssd_space = st.number_input("Espacio en SSD (GB, dejar como 0 si no tiene):", min_value=0, max_value=2048, value=256, step=1)
    weight_kg = st.number_input("Peso del computador (KG):", min_value=0.5, max_value=10.0, value=1.5, step=0.1)

    # Botón para procesar
    if st.button("Procesar"):
        if hdd_space == 0 and ssd_space == 0:
            st.error("HDD y SSD no pueden ser 0 al mismo tiempo.")
        else:
            # Mostrar los datos ingresados
            st.write("### Datos ingresados:")
            st.write(f"- Tipo de Laptop: {tipo_laptop}")
            st.write(f"- Tamaño de la pantalla (Pulgadas): {inches}")
            st.write(f"- Resolución: {ancho} x {alto}")
            st.write(f"- Marca de la GPU: {gpu_brand}")
            st.write(f"- Marca de la CPU: {cpu_brand}")
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
                (knn, rf, lasso_grid_search), input_data, scaler, 
                ['Inches', 'ResolutionNumber', 'RAM_GB', 'HDD_space', 'SSD_space', 'Weight_KG'],
                ['TypeName_Notebook', 'TypeName_Netbook', 'TypeName_Workstation', 'TypeName_Gaming', 'TypeName_2 in 1 Convertible', 'TypeName_Ultrabook', 'GPU_brand_Nvidia', 'GPU_brand_AMD', 'GPU_brand_Intel']
            )

            st.write("### Predicciones del precio del computador:")
            st.write(f"Predicción con KNN: {price_knn:.2f} EUR")
            st.write(f"Predicción con Random Forest: {price_rf:.2f} EUR    -> Esta es la mejor predicción 😉")
            st.write(f"Predicción con Lasso: {price_lasso:.2f} EUR")

# Pestaña de rendimiento de los modelos
with tabs[2]:
    st.header("Rendimiento de los modelos")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    knn, rf, lasso_cv = train_models(X_train, y_train)

    # Predicciones para los tres modelos
    y_pred_knn = knn.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_lasso = lasso_cv.predict(X_test)

    # Generar los gráficos
    fig1, importance_fig, fig_lasso_cv, fig_r2_knn, fig_r2_rf, fig_r2_lasso, fig_rmse = plot_model_performance(df, y_test, y_pred_knn, y_pred_rf, y_pred_lasso, rf, lasso_cv)

    # Mostrar los gráficos en Streamlit
    st.plotly_chart(fig1)
    st.markdown("En este gráfico se pueden comparar los valores reales con los predichos por cada modelo. Se infiere que, precios bajos, los modelos son bastante precisos. Sin embargo, al aumentar el precio de los computadores, los modelos subestiman sus precios.")
    st.plotly_chart(importance_fig)
    st.markdown("En este gráfico de barras, se observa que la memoria RAM y el peso (Weight_KG) son los atributos más importantes para predecir el precio. Es relevante señalar que el peso puede estar correlacionado con computadores más grandes, de escritorio, que tienden a ser mejores y con precios más altos.")
    st.plotly_chart(fig_lasso_cv)
    st.markdown("Se aprecia que el valor de alpha que minimiza el error cuadrático medio es alpha = 0.68")
    st.plotly_chart(fig_r2_knn)
    st.plotly_chart(fig_r2_rf)
    st.plotly_chart(fig_r2_lasso)
    st.markdown("A partir de los R^2 obtenidos a partir de la muestra de validación de los modelos, se aprecia que el que explica más la variabilidad de los precios es Random Forest, con un R^2 de 0.788")
    st.plotly_chart(fig_rmse)
    st.markdown("En los gráficos de barra que indican la raíz del error cuadrático medio, es posible notar que Random Forest es el de menor error, con un valor de 326.41")
    st.markdown(
        """
        Las marcadas diferencias en las predicciones de los modelos para un computador con las mismas variables explicativas podrían explicarse por las 
        limitaciones de la regresión Lasso, que no captura la no linealidad ni las interacciones entre las variables, características que sí pueden ser modeladas 
        por Random Forest y KNN. Además, KNN enfrenta dificultades cuando el computador que se desea predecir se encuentra lejos de los puntos utilizados durante el 
        entrenamiento, ya que no tendrá vecinos suficientemente cercanos para realizar una predicción precisa. Estas razones podrían explicar por qué Random Forest es el 
        mejor modelo entre los tres en este caso.
        """
        )

# Pestaña de análisis exploratorio
with tabs[3]:
    st.header("Análisis Exploratorio")
    st.subheader("Descripción de los Datos")
    
    # Mostrar un resumen de las columnas clave utilizadas
    st.write("**Vista previa de los datos más relevantes:**")
    st.write("""
    Una muestra de las primeras filas de las columnas más relevantes para los modelos.
    """)
    st.code("""
    Inches   ResolutionNumber   RAM_GB   HDD_space   SSD_space   Weight_KG   Price_euros
    15.6            2073600        8          0         256         1.37          977
    13.3            1440000        8          0         128         1.34          1144
    15.6            2073600        8          0         512         2.50          1500
    17.3            2073600       16          0         1024        2.80          2000
    14.0            1920000        4         512          0         1.20           600
    """)

    st.write("**Estadísticas descriptivas de variables clave:**")
    st.code("""
    Precio (€):
        count    5212.000
        mean     1123.687
        std       698.808
        min       174.000
        25%       599.000
        50%       977.000
        75%      1488.990
        max      6099.000

    RAM (GB):
        count    5212.000
        mean       8.764
        std        4.865
        min        2.000
        max      128.000

    SSD (GB):
        count    5212.000
        mean     267.432
        std      333.201
        min        0.000
        max     2048.000
    """)

    # Correlación entre variables
    st.subheader("Correlación entre Variables")
    st.write("""
    A continuación, se muestra la matriz de correlación para identificar las relaciones más relevantes entre las variables y el precio:
    """)

    st.code("""
    Matriz de Correlación:
                     Price_euros  RAM_GB  SSD_space  Weight_KG
    Price_euros          1.000    0.670      0.490      0.150
    RAM_GB               0.670    1.000      0.300      0.050
    SSD_space            0.490    0.300      1.000      0.070
    Weight_KG            0.150    0.050      0.070      1.000
    """)

    st.write("""
    - La RAM tiene una correlación alta con el precio, lo que sugiere que es una variable clave.
    - El SSD también muestra una correlación significativa con el precio.
    - El peso tiene una relación más débil, pero sigue siendo relevante para ciertos tipos de laptops.
    """)

    # Distribución del precio
    st.subheader("Distribución de Precios")
    st.write("""
    Los precios de las laptops se distribuyen principalmente en un rango de €500 a €1500, con algunos modelos de alta gama que superan los €3000.
    """)

    st.code("""
    Rango de precios más común: €500 - €1500
    Laptops de alta gama (>€3000): ~5% del total
    """)

    # Modelos utilizados
    st.subheader("Modelos Utilizados")
    st.write("""
    **1. K-Nearest Neighbors (KNN):**
    - Basado en promediar los resultados de los vecinos más cercanos.
    - Requiere escalado de datos para distancias consistentes.

    **2. Random Forest:**
    - Combina múltiples árboles de decisión para capturar relaciones no lineales.
    - Robusto frente a valores atípicos y datos desbalanceados.

    **3. Regresión LASSO:**
    - Penaliza coeficientes pequeños para evitar sobreespecificar.
    - Selecciona características relevantes en bases de datos de alta dimensionalidad.
    """)

    # Conclusión
    st.subheader("Conclusiones del Análisis Exploratorio")
    st.write("""
    - Las variables **RAM**, **Resolución**, y **SSD** son las más influyentes para predecir el precio.
    - La base de datos está bien equilibrada, con valores suficientes para entrenar modelos robustos.
    - Este análisis respalda la elección de los modelos Random Forest y Lasso para capturar relaciones no lineales y seleccionar variables clave, respectivamente.
    """)

