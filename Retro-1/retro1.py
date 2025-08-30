import pandas as pd
import numpy as np
import matplotlib as plt
import os

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'Student-Performance-1.csv')

data = pd.read_csv(csv_path)

df = pd.DataFrame(data, columns = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index'])

dfTrain = df.iloc[:8000, :]
dfTest = df.iloc[8000:, :]

X = dfTrain[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']].values

def initialize_parameters():
    np.random.seed(1)
    W = np.random.randn(4, 1) * 0.01 
    b = np.zeros((1, 1))
    return W, b


def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))  
    return A

W, b = initialize_parameters()
A = forward_propagation(X, W, b)
print("Final Output:", A)
