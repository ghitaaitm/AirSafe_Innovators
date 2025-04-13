import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Chargement des données en précisant que la première ligne contient les noms de colonnes
data = pd.read_excel('/kaggle/input/yourdata/ClasseurData.xlsx', header=1)

# Affichage des premières lignes pour vérifier les colonnes et les données
print("Premières lignes du fichier :", data.head())

# Vérification de l'existence de la colonne 'Datetime'
if 'Datetime' in data.columns:
    # Conversion de la colonne 'Datetime' en type datetime
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
else:
    raise ValueError("La colonne 'Datetime' est manquante dans le fichier.")

# Remplir les valeurs manquantes par la moyenne des colonnes numériques
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

# Extraction des caractéristiques temporelles
data['year'] = data['Datetime'].dt.year
data['month'] = data['Datetime'].dt.month
data['day'] = data['Datetime'].dt.day
data['hour'] = data['Datetime'].dt.hour

# Suppression de la colonne 'Datetime' après extraction des informations
data.drop(columns='Datetime', inplace=True)

# Vérification des colonnes pour s'assurer que 'AQI' est présente
print("Colonnes disponibles :", data.columns)

# Séparation des données en X (features) et y (target)
if 'AQI' in data.columns:
    # Clip AQI to standard range [0, 500]
    data['AQI'] = np.clip(data['AQI'], 0, 500)
    X = data.drop(columns=['AQI'])
    y = data['AQI']
else:
    raise KeyError("La colonne 'AQI' est manquante dans le fichier.")

# Normalisation des données
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Vérifier la plage de y après normalisation
print("AQI après clipping - min:", y.min(), "max:", y.max())
print("y_scaled - min:", y_scaled.min(), "max:", y_scaled.max())

def create_sequences(X, y, window=30):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

# Création des séquences
X_seq, y_seq = create_sequences(X_scaled, y_scaled)

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)

# Fonction pour construire le modèle CNN-LSTM
def build_cnn_lstm(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# Construction du modèle CNN-LSTM
model = build_cnn_lstm(X_train.shape[1:])
model.compile(optimizer='adam', loss='mse')

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_test, y_test))

# Sauvegarde du modèle
model.save('/kaggle/working/cnn_lstm_model.h5')

# Prédictions et inversions des échelles
y_pred = model.predict(X_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# Visualisation des résultats
plt.plot(y_test_inv, label='Actual AQI')
plt.plot(y_pred_inv, label='Predicted AQI')
plt.legend()
plt.title("AQI Prediction with CNN-LSTM")
plt.ylim(0, 500) 
plt.savefig('/kaggle/working/cnn_lstm_prediction.png')
plt.show()

# Sauvegarde des scalers
joblib.dump(scaler_x, '/kaggle/working/scaler_x.pkl')
joblib.dump(scaler_y, '/kaggle/working/scaler_y.pkl')