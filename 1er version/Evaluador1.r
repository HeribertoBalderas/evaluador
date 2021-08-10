# Inicio del proyecto
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Base de datos
df = pd.read_csv('C:/Users/BALDERFH/Desktop/DATA ANALYSIS/Proyecto Curso/Venta.csv')
df.head()
df["Precio"] = round(df["Valor"] / df["Volumen"])
df.head()
df["Fecha"] = df["Ejercicio"].astype(str) + " " + df["Mes"]
df.head()

# Filtro de datos
cloralex = df["Sku"] == "Cloralex Regular 2lts"
df[cloralex].head()
filtro = df[cloralex]

# Analisis de informacion
filtro.groupby('Mes')['Volumen'].mean()
filtro.groupby('Precio')['Volumen'].mean()

sns.set(style="whitegrid")
sns.boxplot(x=filtro['Volumen'])
plt.axvline(filtro['Volumen'].mean(), c='y')

piezas = filtro["Volumen"]
segmentos = pd.cut(piezas, 20)
filtro['Volumen'].groupby(segmentos).count()

iqr = filtro['Volumen'].quantile(0.75) - filtro['Volumen'].quantile(0.25)
filtro_inferior = filtro['Volumen'] > filtro['Volumen'].quantile(0.25) - (iqr * 1.5)
filtro_superior = filtro['Volumen'] < filtro['Volumen'].quantile(0.75) + (iqr * 1.5)

df_filtrado = filtro[filtro_inferior & filtro_superior]

sns.boxplot(x=df_filtrado['Volumen'])

# Modelo estadistico
series = df_filtrado["Volumen"].to_numpy()
df_filtrado.shape

plt.figure(figsize=(10,6))
plt.plot(series)
plt.xlabel("Fecha")
plt.ylabel("Volumen")
plt.grid(True)

window_size = 5

X = None 
Y = None

for counter in range(len(series)-window_size-1):
    muestra = np.array([series[counter:counter+window_size]])
    salida = np.array([series[counter+window_size]])
    if X is None:
        X = muestra
    else:
        X = np.append(X,muestra,axis=0)
    if Y is None:
        Y = salida
    else:
        Y = np.append(Y,salida)

l0 = tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu")
l1 = tf.keras.layers.Dense(10, activation="relu")
l_output = tf.keras.layers.Dense(1)

model = tf.keras.models.Sequential([l0,l1,l_output])
model.compile(loss="mse",optimizer=tf.keras.optimizers.SGD(lr=1e-6,momentum=0.9),metrics=['mae'])
model.fit(X,Y,epochs=100,batch_size=32,verbose=1,validation_split=0.2)

# Resultado
forecast = []
for time in range(len(series)-window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]

results = np.array(forecast)[:,0,0]

plt.figure(figsize=(10,6))
plt.plot(series,"-")
plt.xlabel("Fecha")
plt.ylabel("Volumen")
plt.grid(True)

plt.plot(results,"-")
plt.xlabel("Fecha")
plt.ylabel("Volumen")
plt.grid(True)
