import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar los datos
current_dir = os.path.dirname(__file__)
train_csv_path = os.path.join(current_dir, 'dataset/train.csv')
test_csv_path = os.path.join(current_dir, 'dataset/test.csv')
valid_csv_path = os.path.join(current_dir, 'dataset/gender_submission.csv')
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)
valid_data = pd.read_csv(valid_csv_path)

# Preparar los datos
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=10000, max_depth=100, random_state=1)
model.fit(X_train, y_train)

# Predicciones
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

# Métricas de desempeño
train_accuracy = accuracy_score(y_train, train_predictions)
val_accuracy = accuracy_score(y_val, val_predictions)
test_accuracy = accuracy_score(valid_data["Survived"], test_predictions)

# Cálculo de sesgo y varianza
bias = 1 - train_accuracy  # Error en el conjunto de entrenamiento
variance = val_accuracy - train_accuracy  # Diferencia entre validación y entrenamiento

# Determinación del nivel de ajuste
if bias > 0.15:  # Sesgo alto
    fit_status = "Underfitting"
elif abs(variance) > 0.1:  # Varianza alta
    fit_status = "Overfitting"
else:
    fit_status = "Fitting"

# Imprimir resultados
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Bias (Error en entrenamiento): {bias:.2f}")
print(f"Variance (Diferencia entre validación y entrenamiento): {variance:.2f}")
print(f"Model Fit Status: {fit_status}")

# Matriz de confusión para el conjunto de prueba
cm = confusion_matrix(valid_data["Survived"], test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Métricas adicionales
precision = precision_score(valid_data["Survived"], test_predictions)
recall = recall_score(valid_data["Survived"], test_predictions)
f1 = f1_score(valid_data["Survived"], test_predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


