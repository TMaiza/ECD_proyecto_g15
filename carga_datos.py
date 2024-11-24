import pandas as pd
import sqlite3

# Cargar datos desde un archivo CSV.
def cargar_datos(filepath):
    return pd.read_csv(filepath)

# Procesar la columna ScreenResolution para extraer Touchscreen y ResolutionNumber.
def procesar_screen_resolution(df):
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'touch' in str(x).lower() else 0)
    df['ResolutionNumber'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')

    def calcular_area(resolution):
        width, height = map(int, resolution.split('x'))
        return width * height

    df['ResolutionNumber'] = df['ResolutionNumber'].apply(calcular_area)
    return df

# Procesar la columna Cpu para extraer CPU_brand.
def procesar_cpu(df):
    df['CPU_brand'] = df['Cpu'].apply(
        lambda x: 'Intel' if 'intel' in x.lower() else ('AMD' if 'amd' in x.lower() else 'Samsung')
    )
    return df

# Procesar la columna Ram para extraer RAM_GB.
def procesar_ram(df):
    df['RAM_GB'] = df['Ram'].str.extract(r'(\d+)').astype(int)
    return df

# Procesar la columna Memory para extraer HDD_space, SSD_space, Flash_space y Hybrid_space.
def procesar_memoria(df):
    df['HDD_space'] = 0
    df['SSD_space'] = 0
    df['Flash_space'] = 0
    df['Hybrid_space'] = 0

    def assign_memory_spaces(memory_str):
        parts = memory_str.split('+')
        for part in parts:
            part = part.strip()
            size_match = pd.Series([part]).str.extract(r'(\d+\.?\d*)\s*(GB|TB)')
            if not size_match.empty:
                size_value = float(size_match[0][0])
                size_unit = size_match[1][0]
                if size_unit == 'TB':
                    size_value *= 1024
                index = df.index[df['Memory'] == memory_str]
                if 'SSD' in part:
                    df.loc[index, 'SSD_space'] = int(size_value)
                elif 'HDD' in part:
                    df.loc[index, 'HDD_space'] = int(size_value)
                elif 'Flash' in part:
                    df.loc[index, 'Flash_space'] = int(size_value)
                elif 'Hybrid' in part:
                    df.loc[index, 'Hybrid_space'] = int(size_value)

    df['Memory'].apply(assign_memory_spaces)
    return df

# Procesar la columna Gpu para extraer GPU_brand.
def procesar_gpu(df):
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
            return 'Unknown'

    df['GPU_brand'] = df['Gpu'].apply(get_gpu_brand)
    return df

# Procesar la columna Weight para convertir a kilogramos.
def procesar_peso(df):
    def convert_weight(weight_str):
        return float(weight_str.replace('kg', '').strip())

    df['Weight_KG'] = df['Weight'].apply(convert_weight)
    return df

# Procesar la columna OpSys para normalizar nombres.
def procesar_op_sys(df):
    def modificar_opsys(opsys):
        if 'Windows' in opsys:
            return 'Windows'
        elif opsys in ['No OS', 'Android', 'Chrome OS']:
            return 'Otro'
        elif opsys in ['Mac OS X', 'macOS']:
            return 'Mac'
        else:
            return opsys

    df['OpSys'] = df['OpSys'].apply(modificar_opsys)
    return df

# Guardar los datos procesados en una base de datos SQLite.
def guardar_datos(df, database_path):
    conn = sqlite3.connect(database_path)
    
    # Guardar todos los datos
    df.to_sql('laptops', conn, if_exists='replace', index=False)

    # Guardar tabla acotada
    columnas_seleccionadas = [
        'Price_euros', 'Product', 'Company', 'TypeName', 'OpSys',
        'Inches', 'Weight_KG', 'Touchscreen', 'ResolutionNumber',
        'GPU_brand', 'CPU_brand', 'RAM_GB', 'HDD_space', 
        'SSD_space', 'Flash_space', 'Hybrid_space'
    ]
    df[columnas_seleccionadas].to_sql('laptops_acotada', conn, if_exists='replace', index=False)

    conn.close()

# Función principal para procesar todos los datos y guardarlos en SQLite.
def procesar_datos(filepath="laptop_price.csv", database_path='laptop_prices_final.sqlite'):

    df = cargar_datos(filepath)
    df = procesar_screen_resolution(df)
    df = procesar_cpu(df)
    df = procesar_ram(df)
    df = procesar_memoria(df)
    df = procesar_gpu(df)
    df = procesar_peso(df)
    df = procesar_op_sys(df)
    guardar_datos(df, database_path)

# Ejecutar la función principal
procesar_datos()
