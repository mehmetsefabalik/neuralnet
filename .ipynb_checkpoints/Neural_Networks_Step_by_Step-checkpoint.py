# Neural Networks Step by Step

# Step 1: Introduction to Neural Networks

# A neural network is a type of machine learning model inspired by the human brain. It is made up of layers of interconnected nodes (neurons), which can learn to recognize patterns in data.

# Why use neural networks?
# - They can learn complex relationships in data.
# - They are used in image recognition, language translation, and more.

# Key Terms
# - Neuron: A single unit in the network that processes input.
# - Layer: A group of neurons. Networks have input, hidden, and output layers.
# - Weights: Parameters that adjust as the network learns.
# - Activation Function: Decides if a neuron should be activated.

# ---

# Step 2: The Perceptron (the simplest neural network)
# Let's build a perceptron to understand the basics.

import numpy as np

# Perceptron function
def perceptron(x, w, b):
    z = np.dot(x, w) + b
    return 1 if z > 0 else 0

# Example inputs
x = np.array([1, 0])  # Input features
w = np.array([0.5, -0.6])  # Weights
b = 0.1  # Bias

output = perceptron(x, w, b)
print(f'Perceptron output: {output}')

# ---

# Step 3: Building a Simple Neural Network
# Let's build a neural network with one hidden layer.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input
X = np.array([[0,0],[0,1],[1,0],[1,1]])
# Output (XOR problem)
y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 2)
B1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
B2 = np.zeros((1, 1))

# Forward pass
def forward(X):
    Z1 = np.dot(X, W1) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + B2
    A2 = sigmoid(Z2)
    return A2

output = forward(X)
print('Network output:\n', output)

# ---

# Step 4: Training a Neural Network
# Training means adjusting weights to minimize error. This is done using backpropagation and a loss function.

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Example loss
y_pred = forward(X)
loss = mse(y, y_pred)
print(f'Loss: {loss}')

# ---

# Step 5: Using a Framework (Keras)
# Let's use Keras to build a neural network easily.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build model
model = keras.Sequential([
    layers.Dense(2, activation='sigmoid', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

print('Predictions:', model.predict(X))

