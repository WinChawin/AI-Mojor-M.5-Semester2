import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_params():
    # He initialization
    W1 = np.random.randn(200, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((200, 1))
    W2 = np.random.randn(10, 200) * np.sqrt(2 / 200)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def relu(z):
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(z, 0.0)

def deriv_relu(z):
    z = np.asarray(z, dtype=np.float64)
    return (z > 0).astype(np.float64)

def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=0, keepdims=True)
    expz = np.exp(z)
    denom = np.sum(expz, axis=0, keepdims=True)
    return expz / denom

def compute_loss(A2, Y, W1, W2, lambd):
    A2 = np.asarray(A2, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int64)
    m = Y.size
    # ป้องกัน log(0)
    eps = 1e-12
    probs_y = np.clip(A2[Y, np.arange(m)], eps, 1.0)
    data_loss = -np.sum(np.log(probs_y)) / m
    reg_loss = (lambd / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2))
    return data_loss + reg_loss

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, lambd):
    # บังคับ dtype ให้หมด
    Z1 = np.asarray(Z1, dtype=np.float64)
    A1 = np.asarray(A1, dtype=np.float64)
    A2 = np.asarray(A2, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    X  = np.asarray(X,  dtype=np.float64)
    Y  = np.asarray(Y,  dtype=np.int64)

    m = Y.size
    one_hot_Y = one_hot(Y, num_classes=W2.shape[0])

    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * np.dot(dZ2, A1.T) + (lambd / m) * W2
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * deriv_relu(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T) + (lambd / m) * W1
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def one_hot(Y, num_classes):
    Y = np.asarray(Y, dtype=np.int64)
    one_hot_Y = np.zeros((num_classes, Y.size), dtype=np.float64)
    one_hot_Y[Y, np.arange(Y.size)] = 1.0
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

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int64)

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
Y_all = data[:, 0].astype(np.int64)
X_all = data[:, 1:].astype(np.float64) / 255.0
X_all = X_all.T
m_total = X_all.shape[1]
X_dev, Y_dev = X_all[:, :1000], Y_all[:1000]
X_train, Y_train = X_all[:, 1000:], Y_all[1000:]


# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=1.0, n_iters=1000, batch_size=64, lambd=0.1)

# Evaluate on the development set
_, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
dev_predictions = get_predictions(A2_dev)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Development Set Accuracy: {dev_accuracy:.4f}")
