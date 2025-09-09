import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Cargar el dataset
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'Student-Performance-1.csv')

data = pd.read_csv(csv_path)

# Preprocesamiento del dataset
df = pd.DataFrame(data, columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 
                                 'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index'])

df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']] = \
    df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

# Dividir en conjunto de entrenamiento y prueba
train, test = train_test_split(df, test_size=0.2, random_state=42)

X_train = train.drop(columns=['Performance Index']).values
y_train = train['Performance Index'].values.reshape(-1, 1)
X_test = test.drop(columns=['Performance Index']).values
y_test = test['Performance Index'].values.reshape(-1, 1)

# Escalar las etiquetas entre 0 y 1
y_min = y_train.min()
y_max = y_train.max()
y_train = (y_train - y_min) / (y_max - y_min)
y_test = (y_test - y_min) / (y_max - y_min)

# Clase de la red neuronal
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        # Inicialización de pesos y sesgos con He
        for i in range(self.num_layers):
            limit = np.sqrt(2 / layer_sizes[i])
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def feed_forward(self, X):
        self.activations = [X]
        for i in range(self.num_layers - 1):  # Capas ocultas
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            z = np.clip(z, -500, 500)  # Limitar valores extremos
            a = self.relu(z)  # Usar ReLU en capas ocultas
            self.activations.append(a)
        # Capa de salida (sin función de activación)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        z = np.clip(z, -500, 500)  # Limitar valores extremos
        self.activations.append(z)
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate):
        deltas = [None] * self.num_layers

        # Delta de la capa de salida
        output_error = y - self.activations[-1]
        deltas[-1] = output_error  # Sin derivada porque la salida es lineal

        # Propagar el error hacia atrás
        for i in range(self.num_layers - 2, -1, -1):
            error = np.dot(deltas[i + 1], self.weights[i + 1].T)
            deltas[i] = error * self.relu_derivative(self.activations[i + 1])

        # Actualizar pesos y sesgos
        for i in range(self.num_layers):
            grad_w = np.dot(self.activations[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)

            # Gradiente clipping para evitar valores extremos
            grad_w = np.clip(grad_w, -1, 1)
            grad_b = np.clip(grad_b, -1, 1)

            self.weights[i] += grad_w * learning_rate
            self.biases[i] += grad_b * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feed_forward(X)
            loss = np.mean(np.square(y - output))  # MSE
            if np.isnan(loss):  # Verificar si el Loss es NaN
                print(f"Loss se volvió NaN en la época {epoch}. Deteniendo el entrenamiento.")
                break
            print(f"Epoch {epoch}, Loss: {loss}")
            self.backward(X, y, learning_rate)

# Entrenamiento y evaluación
if __name__ == "__main__":
    # Crear la red neuronal con más capas ocultas
    layer_sizes = [X_train.shape[1], 16, 32, 16, 1]  # Más neuronas en las capas ocultas
    nn = NeuralNetwork(layer_sizes)

    # Entrenar la red neuronal
    nn.train(X_train, y_train, epochs=10000, learning_rate=0.005)

    # Realizar predicciones
    predictions = nn.feed_forward(X_test)

    # Desescalar las predicciones
    predictions = predictions * (y_max - y_min) + y_min

    # Evaluar el modelo
    mse = np.mean(np.square(y_test * (y_max - y_min) + y_min - predictions))
    print(f"Mean Squared Error (MSE) en el conjunto de prueba: {mse:.2f}")