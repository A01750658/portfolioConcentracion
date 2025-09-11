import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'Student-Performance-1.csv')

data = pd.read_csv(csv_path)

df = pd.DataFrame(data, columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 
                                 'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index'])

df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']] = \
    df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

train, test = train_test_split(df, test_size=0.2, random_state=42)

X_train = train.drop(columns=['Performance Index']).values
y_train = train['Performance Index'].values.reshape(-1, 1)
X_test = test.drop(columns=['Performance Index']).values
y_test = test['Performance Index'].values.reshape(-1, 1)

y_min = y_train.min()
y_max = y_train.max()
y_train = (y_train - y_min) / (y_max - y_min)
y_test = (y_test - y_min) / (y_max - y_min)

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
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            z = np.clip(z, -500, 500)  
            a = self.relu(z)
            self.activations.append(a)
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        z = np.clip(z, -500, 500)  
        self.activations.append(z)
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate):
        deltas = [None] * self.num_layers

        
        output_error = y - self.activations[-1]
        deltas[-1] = output_error  

        
        for i in range(self.num_layers - 2, -1, -1):
            error = np.dot(deltas[i + 1], self.weights[i + 1].T)
            deltas[i] = error * self.relu_derivative(self.activations[i + 1])

        
        for i in range(self.num_layers):
            grad_w = np.dot(self.activations[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)

            
            grad_w = np.clip(grad_w, -1, 1)
            grad_b = np.clip(grad_b, -1, 1)

            self.weights[i] += grad_w * learning_rate
            self.biases[i] += grad_b * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feed_forward(X)
            loss = np.mean(np.square(y - output))  
            if np.isnan(loss):  
                print(f"Loss se volvió NaN en la época {epoch}. Deteniendo el entrenamiento.")
                break
            print(f"Epoch {epoch}, Loss: {loss}")
            self.backward(X, y, learning_rate)


if __name__ == "__main__":
    
    layer_sizes = [X_train.shape[1], 16, 32, 16, 1]  
    nn = NeuralNetwork(layer_sizes)

    
    nn.train(X_train, y_train, epochs=10000, learning_rate=0.001)

    
    predictions = nn.feed_forward(X_test)

    
    y_pred = predictions * (y_max - y_min) + y_min

    y_pred_classes = (predictions > 0.5).astype(int)
    y_test_classes = (y_test > 0.5).astype(int)

    cm = confusion_matrix(y_test_classes, y_pred_classes)

    sns.heatmap(cm, annot=True, fmt = "d", cmap = "magma", xticklabels = ['Failed', 'Passed'], yticklabels=['Failed', 'Passed'])
    plt.title("Confusion Matrix")
    plt.show()
    
    y_test_norm = y_test*(y_max - y_min) + y_min
    
    #y_test * (y_max - y_min) + y_min - y_pred
    mse = np.mean(np.square(y_test_norm - y_pred))
    print(f"Mean Squared Error (MSE) en el conjunto de prueba: {mse:.2f}")