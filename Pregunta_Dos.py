import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

# ---------------------------- CARGA DEL ARCHIVO CSV ----------------------------
# Cargar el archivo CSV proporcionado por el usuario
file_path = r'C:\Users\s2dan\OneDrive\Documentos\WorkSpace\PrimerParcia_IA\ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)
##### La columna NObeyesdad sería la clase del DataSet ObesityDataSet_raw_and_data_sinthetic.csv #####
# Descripción general del conjunto de datos:
# Edad: Edad del individuo.
# Género: Género del individuo.
# Altura: Altura del individuo.
# Peso: Peso del individuo.
# CALC: Ingesta calórica.
# FAVC: Consumo frecuente de alimentos con alto contenido calórico.
# FCVC: Frecuencia de consumo de verduras.
# NCP: Número de comidas principales.
# SCC: Consumo de bebidas dulces.
# HUMO: Hábito tabáquico.
# CH2O: Ingesta diaria de agua.
# Antecedentes familiares de sobrepeso: si hay antecedentes familiares de sobrepeso.
# FAF: Frecuencia de actividad física.
# MAR: Tiempo utilizando dispositivos tecnológicos.
# CAEC: Consumo de alimentos entre comidas.
# MTRANS: Método de transporte.
# NObeyesdad: Nivel de obesidad.        *****CLASE*****


# ---------------------------- INCISO A ----------------------------
# Funciones para calcular percentiles y cuartiles manualmente para columnas numéricas
def calculate_percentile(data, percentile):
    sorted_data = sorted(data)
    n = len(sorted_data)
    k = (n - 1) * percentile / 100
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_data[f] + (sorted_data[f + 1] - sorted_data[f]) * c
    else:
        return sorted_data[f]

def calculate_quartiles(data):
    q1 = calculate_percentile(data, 25)
    q2 = calculate_percentile(data, 50)  # Esto es la mediana
    q3 = calculate_percentile(data, 75)
    return q1, q2, q3

# Función para calcular frecuencias para columnas no numéricas
def calculate_frequencies(column_data):
    frequency_dict = {}
    for item in column_data:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1
    return frequency_dict

# Filtrar columnas numéricas
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
manual_results = {}

# Cálculo de percentiles y cuartiles para cada columna numérica
for column in numeric_columns:
    column_data = data[column].dropna().tolist()  # Ignorar valores nulos si los hay
    if column_data:  # Solo procesar columnas que tengan datos
        percentile_90 = calculate_percentile(column_data, 90)
        q1, q2, q3 = calculate_quartiles(column_data)
        manual_results[column] = {
            "Percentil 90": percentile_90,
            "Cuartil 1": q1,
            "Mediana (Cuartil 2)": q2,
            "Cuartil 3": q3
        }

        # Graficar el histograma para las columnas numéricas
        plt.figure(figsize=(6, 4))
        plt.hist(column_data, bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Distribución de {column}')
        plt.xlabel(f'{column}')
        plt.ylabel('Frecuencia')
        plt.show()

        # Análisis de la forma de la distribución
        mean_value = sum(column_data) / len(column_data)
        median_value = calculate_percentile(column_data, 50)
        
        if abs(mean_value - median_value) < (0.1 * mean_value):
            print(f"  La columna '{column}' parece seguir una distribución normal (Gaussiana).")
        else:
            print(f"  La columna '{column}' no parece seguir una distribución normal.")
        
        if set(column_data) == {0, 1}:
            print(f"  La columna '{column}' puede seguir una distribución Bernoulli.")
        elif min(column_data) >= 0 and all(isinstance(x, int) for x in column_data):
            print(f"  La columna '{column}' puede seguir una distribución de Poisson (conteo de eventos).")
        print()

# Mostrar resultados de percentiles y cuartiles para columnas numéricas
print("Resultados de Percentiles y Cuartiles para columnas numéricas:")
for column, result in manual_results.items():
    print(f"\nColumna: {column}")
    for key, value in result.items():
        print(f"  {key}: {value}")

# Filtrar columnas no numéricas
non_numeric_columns = data.select_dtypes(include=['object', 'bool']).columns

# Cálculo de frecuencias para columnas no numéricas
for column in non_numeric_columns:
    column_data = data[column].dropna().tolist()  # Ignorar valores nulos si los hay
    
    if column_data:  # Si la columna no está vacía
        print(f"\nFrecuencia de valores en la columna '{column}':")
        frequencies = calculate_frequencies(column_data)
        
        # Mostrar las frecuencias de cada categoría
        for value, freq in frequencies.items():
            print(f"  {value}: {freq} veces")
        
        # Graficar un histograma de barras para las frecuencias
        plt.figure(figsize=(8, 4))
        plt.bar(frequencies.keys(), frequencies.values(), color='lightblue', edgecolor='black')
        plt.title(f"Distribución de {column}")
        plt.xlabel(f"{column}")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45)  # Rotar etiquetas si es necesario
        plt.show()

        # Análisis de distribuciones
        unique_values = set(column_data)
        if len(unique_values) == 2:  # Caso binario (Bernoulli)
            print(f"  La columna '{column}' puede seguir una distribución Bernoulli.")
        else:
            print(f"  La columna '{column}' puede seguir una distribución multinomial.")

# ---------------------------- INCISO B ----------------------------
# Selección de 3 columnas: 'Age', 'Height', 'Weight' para análisis más detallado

# Características relevantes para cada columna

# 1. Edad (Age)
print("Características de la columna 'Age':")
age_mean = data['Age'].mean()
age_median = data['Age'].median()
age_min = data['Age'].min()
age_max = data['Age'].max()
age_std = data['Age'].std()

print(f"- Rango de valores: {age_min} - {age_max}")
print(f"- Media: {age_mean:.2f}")
print(f"- Mediana: {age_median:.2f}")
print(f"- Desviación estándar (varianza): {age_std:.2f}")

# 2. Altura (Height)
print("\nCaracterísticas de la columna 'Height':")
height_mean = data['Height'].mean()
height_median = data['Height'].median()
height_min = data['Height'].min()
height_max = data['Height'].max()
height_std = data['Height'].std()

print(f"- Rango de valores: {height_min} - {height_max}")
print(f"- Media: {height_mean:.2f}")
print(f"- Mediana: {height_median:.2f}")
print(f"- Desviación estándar (varianza): {height_std:.2f}")

# 3. Peso (Weight)
print("\nCaracterísticas de la columna 'Weight':")
weight_mean = data['Weight'].mean()
weight_median = data['Weight'].median()
weight_min = data['Weight'].min()
weight_max = data['Weight'].max()
weight_std = data['Weight'].std()

print(f"- Rango de valores: {weight_min} - {weight_max}")
print(f"- Media: {weight_mean:.2f}")
print(f"- Mediana: {weight_median:.2f}")
print(f"- Desviación estándar (varianza): {weight_std:.2f}")

# Graficar diagramas de dispersión para ver las relaciones entre las columnas seleccionadas
plt.figure(figsize=(10, 5))

# Gráfico de dispersión Edad vs Altura
plt.scatter(data['Age'], data['Height'], color='blue', label='Edad vs Altura')
plt.xlabel('Edad')
plt.ylabel('Altura (m)')
plt.title('Dispersión: Edad vs Altura')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))

# Gráfico de dispersión Edad vs Peso
plt.scatter(data['Age'], data['Weight'], color='green', label='Edad vs Peso')
plt.xlabel('Edad')
plt.ylabel('Peso (kg)')
plt.title('Dispersión: Edad vs Peso')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))

# Gráfico de dispersión Altura vs Peso
plt.scatter(data['Height'], data['Weight'], color='red', label='Altura vs Peso')
plt.xlabel('Altura (m)')
plt.ylabel('Peso (kg)')
plt.title('Dispersión: Altura vs Peso')
plt.legend()
plt.show()

# Mapa de calor con matplotlib (sin seaborn)
def plot_correlation_matrix(df, columns):
    corr = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cax = ax.matshow(corr, cmap='coolwarm')
    plt.title('Mapa de Calor: Correlación entre Edad, Altura y Peso', pad=20)
    
    plt.xticks(np.arange(len(columns)), columns, rotation=45)
    plt.yticks(np.arange(len(columns)), columns)
    
    fig.colorbar(cax)
    
    plt.show()

# Llamar a la función para graficar el mapa de calor
plot_correlation_matrix(data, ['Age', 'Height', 'Weight'])
# Gráficos de dispersión:
# Los gráficos de dispersión muestran las relaciones entre las tres columnas seleccionadas: 
# Edad vs Altura, Edad vs Peso, y Altura vs Peso.

# Mapa de calor (opcional):
# Un mapa de calor es una forma de visualizar la correlación entre varias variables. 
# En este caso, muestra la correlación entre Edad, Altura, y Peso.

# Los valores en el mapa de calor oscilan entre -1 y 1:
# - Un valor cercano a 1 indica una fuerte correlación positiva (cuando una variable aumenta, la otra también).
# - Un valor cercano a -1 indica una fuerte correlación negativa (cuando una variable aumenta, la otra disminuye).


# ---------------------------- INCISO C ----------------------------
# Cálculo de media, mediana y moda para las columnas seleccionadas usando pandas
print()
print("Resultados de Media, Mediana y Moda (Inciso C)")

# Seleccionar tres columnas para el análisis
columns_to_analyze = ['Age', 'Height', 'Weight']

# Cálculo de media, mediana y moda para cada columna
for column in columns_to_analyze:
    mean_value = data[column].mean()
    median_value = data[column].median()
    mode_value = data[column].mode()[0]

    print(f"Resultados para '{column}':")
    print(f"- Media: {mean_value:.2f}")
    print(f"- Mediana: {median_value:.2f}")
    print(f"- Moda: {mode_value:.2f}")
    print()

# Graficar un diagrama de cajas y bigotes para las columnas seleccionadas
plt.figure(figsize=(10, 6))

plt.boxplot([data['Age'].dropna(), data['Height'].dropna(), data['Weight'].dropna()], 
            tick_labels=['Edad', 'Altura', 'Peso'])
plt.title('Diagrama de Cajas y Bigotes para Edad, Altura y Peso')
plt.ylabel('Valores')
plt.show()

# Explicación de los resultados del diagrama de cajas y bigotes:
# Cajas (cuartiles):
# - La parte inferior de la caja representa el primer cuartil (Q1), es decir, el valor debajo del cual se encuentra el 25% de los datos.
# - La línea dentro de la caja representa la mediana (Q2), que indica el valor central de los datos.
# - La parte superior de la caja representa el tercer cuartil (Q3), es decir, el valor debajo del cual
