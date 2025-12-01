import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_params(layer1_size=256, layer2_size=128):
    W1 = np.random.randn(layer1_size, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((layer1_size, 1))
    W2 = np.random.randn(layer2_size, layer1_size) * np.sqrt(2 / layer1_size)
    b2 = np.zeros((layer2_size, 1))
    W3 = np.random.randn(10, layer2_size) * np.sqrt(2 / layer2_size)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def relu(z):
    return np.maximum(z, 0)

def deriv_relu(z):
    return z > 0

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def compute_loss(A3, Y, W1, W2, W3, lambd):
    m = Y.size
    log_probs = -np.log(A3[Y, np.arange(m)] + 1e-10)
    data_loss = np.sum(log_probs) / m
    reg_loss = (lambd / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    return data_loss + reg_loss

def back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, lambd):
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    dZ3 = A3 - one_hot_Y
    dW3 = (1/m) * np.dot(dZ3, A2.T) + (lambd / m) * W3
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    dZ2 = np.dot(W3.T, dZ3) * deriv_relu(Z2)
    dW2 = (1/m) * np.dot(dZ2, A1.T) + (lambd / m) * W2
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * deriv_relu(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T) + (lambd / m) * W1
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / len(Y)

def gradient_descent(X, Y, X_val, Y_val, alpha=0.7, n_iters=800, batch_size=128, lambd=0.005):
    W1, b1, W2, b2, W3, b3 = init_params()
    m = X.shape[1]
    
    best_val_accuracy = 0
    best_params = None

    for i in range(n_iters + 1):
        indices = np.random.permutation(m)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]
        
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]

            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, 
                                                       X_batch, Y_batch, lambd)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, 
                                                     dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 10 == 0:
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
            predictions = get_predictions(A3)
            train_accuracy = get_accuracy(predictions, Y)
            loss = compute_loss(A3, Y, W1, W2, W3, lambd)
            
            Z1_val, A1_val, Z2_val, A2_val, Z3_val, A3_val = forward_prop(W1, b1, W2, b2, W3, b3, X_val)
            val_predictions = get_predictions(A3_val)
            val_accuracy = get_accuracy(val_predictions, Y_val)
            
            print(f"Iter {i:4d} | Loss: {loss:.4f} | Train Accuracy: {train_accuracy*100:5.2f}% | Val Accuracy: {val_accuracy*100:5.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())

        if i > 0 and i % 150 == 0:
            alpha *= 0.85

    return best_params if best_params else (W1, b1, W2, b2, W3, b3)

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)

# Load data
data = pd.read_csv('digit-recognizer/train.csv').to_numpy()
np.random.shuffle(data)

# Split data
data_dev = data[:1000].T
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:] / 255.0

data_train = data[1000:].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:] / 255.0

# Train model
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, X_dev, Y_dev)

# Evaluate
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"\nDevelopment Set Accuracy: {dev_accuracy*100:.2f}%")