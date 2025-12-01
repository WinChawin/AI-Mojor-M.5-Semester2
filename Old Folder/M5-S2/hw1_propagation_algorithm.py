import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions=None, learning_rate=0.1, regularization=0.01):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Set default activation functions
        if activation_functions is None:
            self.activation_functions = ['relu'] * (self.num_layers - 2) + ['softmax']
        else:
            self.activation_functions = activation_functions
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        self._init_params()
        
        # Store training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'iterations': []
        }
     
    def _init_params(self):
        """Initialize weights and biases"""
        for i in range(1, self.num_layers):
            self.weights[f'W{i}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(2 / self.layer_sizes[i-1])
            self.biases[f'b{i}'] = np.zeros((self.layer_sizes[i], 1))
    
    def _relu(self, z):
        return np.maximum(z, 0)
    
    def _relu_derivative(self, z):
        return z > 0
    
    def _tanh(self, z):
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow
    
    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _softmax(self, z):
        z = z - np.max(z, axis=0, keepdims=True)  # Numerical stability
        return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
    
    def _apply_activation(self, z, activation):
        """Apply activation function"""
        if activation == 'relu':
            return self._relu(z)
        elif activation == 'tanh':
            return self._tanh(z)
        elif activation == 'sigmoid':
            return self._sigmoid(z)
        elif activation == 'softmax':
            return self._softmax(z)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _apply_activation_derivative(self, z, activation):
        """Apply activation function derivative"""
        if activation == 'relu':
            return self._relu_derivative(z)
        elif activation == 'tanh':
            return self._tanh_derivative(z)
        elif activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        else:
            raise ValueError(f"Unsupported activation derivative for: {activation}")
    
    def _one_hot_encode(self, Y):
        """Convert labels to one-hot encoding"""
        num_classes = self.layer_sizes[-1]
        one_hot_Y = np.zeros((num_classes, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y
    
    def forward_propagation(self, X):
        """Forward propagation through the network"""
        activations = {'A0': X}
        z_values = {}
        
        for i in range(1, self.num_layers):
            # Linear transformation
            z = np.dot(self.weights[f'W{i}'], activations[f'A{i-1}']) + self.biases[f'b{i}']
            z_values[f'Z{i}'] = z
            
            # Apply activation function
            a = self._apply_activation(z, self.activation_functions[i-1])
            activations[f'A{i}'] = a
        
        return activations, z_values
    
    def compute_loss(self, Y_pred, Y_true):
        """Compute cross-entropy loss with L2 regularization"""
        m = Y_true.size
        
        # Cross-entropy loss
        one_hot_Y = self._one_hot_encode(Y_true)
        log_probs = -np.log(Y_pred[Y_true, np.arange(m)] + 1e-8)
        data_loss = np.sum(log_probs) / m
        
        # L2 regularization
        reg_loss = 0
        for i in range(1, self.num_layers):
            reg_loss += np.sum(self.weights[f'W{i}']**2)
        reg_loss = (self.regularization / (2 * m)) * reg_loss
        
        return data_loss + reg_loss
    
    def backward_propagation(self, activations, z_values, X, Y):
        """Backward propagation through the network"""
        m = Y.size
        gradients = {}
        one_hot_Y = self._one_hot_encode(Y)
        
        # Output layer gradient
        dZ = activations[f'A{self.num_layers-1}'] - one_hot_Y
        
        # Backward pass through all layers
        for i in range(self.num_layers-1, 0, -1):
            # Compute gradients for weights and biases
            gradients[f'dW{i}'] = (1/m) * np.dot(dZ, activations[f'A{i-1}'].T) + (self.regularization/m) * self.weights[f'W{i}']
            gradients[f'db{i}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            # Compute gradient for previous layer (if not input layer)
            if i > 1:
                dA_prev = np.dot(self.weights[f'W{i}'].T, dZ)
                dZ = dA_prev * self._apply_activation_derivative(z_values[f'Z{i-1}'], self.activation_functions[i-2])
        
        return gradients
    
    def update_parameters(self, gradients):
        """Update weights and biases using gradients"""
        for i in range(1, self.num_layers):
            self.weights[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']
            self.biases[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']
    
    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[f'A{self.num_layers-1}'], axis=0)
    
    def get_accuracy(self, X, Y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.sum(predictions == Y) / len(Y)
    
    def fit(self, X_train, Y_train, epochs=1000, batch_size=64, verbose=True, plot=True):
        """Train the neural network"""
        m = X_train.shape[1]
        
        for epoch in range(epochs + 1):
            # Mini-batch gradient descent
            for j in range(0, m, batch_size):
                X_batch = X_train[:, j:j+batch_size]
                Y_batch = Y_train[j:j+batch_size]
                
                # Forward and backward propagation
                activations, z_values = self.forward_propagation(X_batch)
                gradients = self.backward_propagation(activations, z_values, X_batch, Y_batch)
                self.update_parameters(gradients)
            
            # Monitor performance
            if epoch % 10 == 0:
                activations, _ = self.forward_propagation(X_train)
                loss = self.compute_loss(activations[f'A{self.num_layers-1}'], Y_train)
                accuracy = self.get_accuracy(X_train, Y_train)
                
                self.training_history['loss'].append(loss)
                self.training_history['accuracy'].append(accuracy)
                self.training_history['iterations'].append(epoch)
                
                if verbose:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
            
            # Learning rate decay
            self.learning_rate = self.learning_rate * (1 / (1 + 0.001 * epoch))
        
        if plot:
            self.plot_training_history()
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['iterations'], self.training_history['accuracy'], label="Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.grid()
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['iterations'], self.training_history['loss'], label="Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print network summary"""
        print("Neural Network Summary:")
        print("=" * 50)
        print(f"Number of layers: {self.num_layers - 1}")
        print(f"Layer sizes: {self.layer_sizes}")
        print(f"Activation functions: {self.activation_functions}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Regularization: {self.regularization}")
        print("=" * 50)
        

# Example usage:
if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv('digit-recognizer/train.csv').to_numpy()
    np.random.shuffle(data)
    
    m, n = data.shape
    data_dev = data[:1000].T
    Y_dev = data_dev[0].astype(int)
    X_dev = data_dev[1:] / 255.0
    
    data_train = data[1000:].T
    Y_train = data_train[0].astype(int)
    X_train = data_train[1:] / 255.0
    
    nn = NeuralNetwork(
        layer_sizes=[784, 100, 10],
        activation_functions=['relu', 'relu', 'softmax'],
        learning_rate=0.1,
        regularization=0.01
    )
    
    # Print network summary
    nn.summary()
    
    # Train the model
    nn.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=True)
    
    # Evaluate on development set
    dev_accuracy = nn.get_accuracy(X_dev, Y_dev)
    print(f"\nDevelopment Set Accuracy: {dev_accuracy*100:.2f}%")