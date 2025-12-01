import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 

def init_params():
    # He initialization
    W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(10, 128) * np.sqrt(2 / 128)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def relu(z):
    return np.maximum(z, 0)

def deriv_relu(z):
    return z > 0

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)  # Numerical stability
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

def compute_loss(A2, Y, W1, W2, lambd):
    m = Y.size
    one_hot_Y = one_hot(Y)
    log_probs = -np.log(A2[Y, np.arange(m)] + 1e-8)  # เพิ่ม epsilon เพื่อป้องกัน log(0)
    data_loss = np.sum(log_probs) / m
    # Add L2 regularization
    reg_loss = (lambd / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2))
    return data_loss + reg_loss

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, lambd):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * np.dot(dZ2, A1.T) + (lambd / m) * W2
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1)
    dW1 = 1/m * np.dot(dZ1, X.T) + (lambd / m) * W1
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def one_hot(Y):
    num_classes = 10  # แก้ให้เป็น 10 คลาสสำหรับ digits 0-9
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / len(Y)

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def gradient_descent(X, Y, alpha, n_iters, batch_size, lambd):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]
    accuracy_list = []
    loss_list = []
    iteration_list = []

    for i in range(n_iters + 1):
        # Mini-batch gradient descent
        for j in range(0, m, batch_size):
            X_batch = X[:, j:j+batch_size]
            Y_batch = Y[j:j+batch_size]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch, lambd)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Monitor performance
        if i % 10 == 0:
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            loss = compute_loss(A2, Y, W1, W2, lambd)
            accuracy_list.append(accuracy)
            loss_list.append(loss)
            iteration_list.append(i)
            print(f"Iteration {i}, Loss: {loss:.4f}, Accuracy: {(accuracy*100):.4f}")

        # Learning rate decay
        alpha = alpha * (1 / (1 + 0.001 * i))

    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iteration_list, accuracy_list, label="Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Iterations")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iteration_list, loss_list, label="Loss", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss over Iterations")
    plt.grid()
    plt.legend()

    plt.show()

    return W1, b1, W2, b2

# Load and preprocess data
data = pd.read_csv('digit-recognizer/train.csv').to_numpy()
np.random.shuffle(data)

m, n = data.shape
data_dev = data[:1000].T
Y_dev = data_dev[0].astype(int)  # แปลงเป็น int
X_dev = data_dev[1:] / 255.0

data_train = data[1000:].T
Y_train = data_train[0].astype(int)  # แปลงเป็น int
X_train = data_train[1:] / 255.0

# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.5, n_iters=1000, batch_size=64, lambd=0.1)

# Evaluate on the development set - แก้ไขตรงนี้
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_dev)
dev_predictions = get_predictions(A2)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Development Set Accuracy: {dev_accuracy:.4f}")