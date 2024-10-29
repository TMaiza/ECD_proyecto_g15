# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3

df = pd.read_csv('laptop_price.csv')

#DESGLOSE INFO DE RESOLUCÍON

# Crear columna 'Touchscreen', buscando touch en ScreenResolution
# 0 no es tocuh, 1 si es touch
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'touch' in str(x).lower() else 0)
# Crear columna 'ResolutionNumber', 
df['ResolutionNumber'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')

#DESGLOSE INFO DE CPU

# Crear la columna 'CPU_brand', extrayendo 'Intel' o 'AMD', aplicando Samsung
# a las que no son ninguna de las 2 anteriores
df['CPU_brand'] = df['Cpu'].apply(lambda x: 'Intel' if 'intel' in x.lower() else ('AMD' if 'amd' in x.lower() else 'Samsung'))

#DESGLOSE INFO DE RAM

# Crear columna 'RAM_GB', extrayendo los dígitos de la columna 'Ram
# y luego convirtiéndolos a enteros
df['RAM_GB'] = df['Ram'].str.extract(r'(\d+)')
df['RAM_GB'] = df['RAM_GB'].astype(int)

#DESGLOSE INFO DE MEMORIA

# Paso 1: Inicializar las columnas HDD_space, SSD_space y Flash_space
df['HDD_space'] = 0
df['SSD_space'] = 0
df['Flash_space'] = 0
df['Hybrid_space'] = 0

# Paso 2: Función para separar y asignar el tamaño a las columnas
def assign_memory_spaces(memory_str):
    # Separar en partes si hay un '+'
    parts = memory_str.split('+')
    for part in parts:
        part = part.strip()  # Limpiar espacios
        # Extraer el tamaño y tipo usando una expresión regular
        size_match = pd.Series([part]).str.extract(r'(\d+\.?\d*)\s*(GB|TB)')  
        
        if not size_match.empty:
            size_value = float(size_match[0][0])  # Extraer valor numérico
            size_unit = size_match[1][0]  # Extraer unidad (GB o TB)
            
            # Normalizar el tamaño a GB
            if size_unit == 'TB':
                size_value *= 1024  # Convertir TB a GB
            
            # Asignar valores a las columnas correspondientes
            index = df.index[df['Memory'] == memory_str]  # Encuentra todos los índices
            if 'SSD' in part:
                df.loc[index, 'SSD_space'] = int(size_value)  # Asignar valor directo
            elif 'HDD' in part:
                df.loc[index, 'HDD_space'] = int(size_value)  # Asignar valor directo
            elif 'Flash' in part:
                df.loc[index, 'Flash_space'] = int(size_value)  # Asignar valor directo
            elif 'Hybrid' in part:
                df.loc[index, 'Hybrid_space'] = int(size_value)  # Asignar valor directo

# Aplicar la función al DataFrame
df['Memory'].apply(assign_memory_spaces)

#DESGLOSE DE GPU

def get_gpu_brand(gpu_str):
    if 'Nvidia' in gpu_str:
        return 'Nvidia'
    elif 'AMD' in gpu_str:
        return 'AMD'
    elif 'Intel' in gpu_str:
        return 'Intel'
    elif 'ARM' in gpu_str:
        return 'ARM'
    else:
        return 'Unknown'  # O cualquier otra etiqueta que quieras usar

# Crear la nueva columna 'GPU_brand'
df['GPU_brand'] = df['Gpu'].apply(get_gpu_brand)

#DESGLOSE WEIGHT

def convert_weight(weight_str):
    # Limpiar la cadena y convertir a float
    return float(weight_str.replace('kg', '').strip())

# Aplicar la función a la columna 'Weight'
df['Weight_KG'] = df['Weight'].apply(convert_weight)


# Conectar a la base de datos SQLite
conn = sqlite3.connect('laptop_prices.sqlite')

# Guardar los datos en SQLite, respetando la estructura existente
df.to_sql('laptops', conn, if_exists='append', index=False)

# Seleccionar columnas para tabla acotada
l = [
     'Price_euros',
    'Product', 
    'Company', 
    'TypeName', 
    'OpSys', 
    'Inches', 
    'Weight_KG', 
    'Touchscreen', 
    'ResolutionNumber', 
    'GPU_brand',
    'CPU_brand', 
    'RAM_GB', 
    'HDD_space', 
    'SSD_space', 
    'Flash_space', 
    'Hybrid_space'
]

columnas_seleccionadas = df[l]

# Crear la nueva tabla en la base de datos SQLite
columnas_seleccionadas.to_sql('laptops_acotada', conn, if_exists='replace', index=False)

# Cerrar la conexión
conn.close()
