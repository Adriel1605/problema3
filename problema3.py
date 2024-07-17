import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos y asignar nombres a las columnas
columnas = [
    'direccion_viento', 'velocidad_viento', 'humedad', 
    'temperatura', 'lluvia', 'presion_atmosferica', 
    'potencia', 'intensidad_luminica'
]
datos_clima = pd.read_csv('datosclima_verano2016.csv', header=None, names=columnas)

# Seleccionar las columnas relevantes para X
X = datos_clima[['velocidad_viento', 'humedad', 'temperatura', 'presion_atmosferica']]

# Leer el archivo saturacion.csv y renombrar su columna
saturacion = pd.read_csv('saturacion.csv', header=None, names=['punto_de_rocio'])
Y = saturacion['punto_de_rocio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(X, Y, test_size=0.2, random_state=42)

# Crear el modelo MLPRegressor
modelo = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Entrenar el modelo
modelo.fit(X_entrenamiento, Y_entrenamiento)

# Predecir y evaluar el modelo
Y_pred_entrenamiento = modelo.predict(X_entrenamiento)
Y_pred_prueba = modelo.predict(X_prueba)

# Calcular las métricas de desempeño
entrenamiento_mse = mean_squared_error(Y_entrenamiento, Y_pred_entrenamiento)
prueba_mse = mean_squared_error(Y_prueba, Y_pred_prueba)
entrenamiento_r2 = r2_score(Y_entrenamiento, Y_pred_entrenamiento)
prueba_r2 = r2_score(Y_prueba, Y_pred_prueba)

print(f'Error Cuadrático Medio en Entrenamiento: {entrenamiento_mse}')
print(f'Error Cuadrático Medio en Prueba: {prueba_mse}')
print(f'R2 en Entrenamiento: {entrenamiento_r2}')
print(f'R2 en Prueba: {prueba_r2}')

# Verificar con las entradas específicas
entradas_verificacion = pd.DataFrame({
    'velocidad_viento': [2.2],
    'humedad': [63.0],
    'temperatura': [22.11],
    'presion_atmosferica': [29.90]
})

prediccion_verificacion = modelo.predict(entradas_verificacion)
print('Predicción para las entradas de verificación:', prediccion_verificacion)

# Comparar con el valor real de punto de rocío de 14.72
valor_real = 14.72
diferencia = abs(prediccion_verificacion[0] - valor_real)
print(f'Diferencia con el valor real: {diferencia}')
