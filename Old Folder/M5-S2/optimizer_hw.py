import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

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
    
    def fit_with_gradient_type(self, X_train, Y_train, X_dev, Y_dev, gradient_type='mini-batch', epochs=100, batch_size=64, verbose=True):
        """Train the neural network with specified gradient descent type"""
        m = X_train.shape[1]
        
        # Store training history
        history = {
            'loss': [],
            'accuracy': [],
            'dev_accuracy': [],
            'epochs': [],
            'times': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs + 1):
            epoch_start = time.time()
            
            if gradient_type == 'batch':
                # Batch Gradient Descent - use entire dataset
                activations, z_values = self.forward_propagation(X_train)
                gradients = self.backward_propagation(activations, z_values, X_train, Y_train)
                self.update_parameters(gradients)
                
            elif gradient_type == 'stochastic':
                # Stochastic Gradient Descent - one sample at a time
                indices = np.random.permutation(m)
                for i in indices:
                    X_sample = X_train[:, i:i+1]
                    Y_sample = Y_train[i:i+1]
                    activations, z_values = self.forward_propagation(X_sample)
                    gradients = self.backward_propagation(activations, z_values, X_sample, Y_sample)
                    self.update_parameters(gradients)
                    
            else:  # mini-batch
                # Mini-batch Gradient Descent
                indices = np.random.permutation(m)
                X_shuffled = X_train[:, indices]
                Y_shuffled = Y_train[indices]
                
                for j in range(0, m, batch_size):
                    X_batch = X_shuffled[:, j:j+batch_size]
                    Y_batch = Y_shuffled[j:j+batch_size]
                    
                    activations, z_values = self.forward_propagation(X_batch)
                    gradients = self.backward_propagation(activations, z_values, X_batch, Y_batch)
                    self.update_parameters(gradients)
            
            epoch_time = time.time() - epoch_start
            
            # Monitor performance every 5 epochs for faster feedback
            if epoch % 5 == 0:
                activations, _ = self.forward_propagation(X_train)
                loss = self.compute_loss(activations[f'A{self.num_layers-1}'], Y_train)
                train_accuracy = self.get_accuracy(X_train, Y_train)
                dev_accuracy = self.get_accuracy(X_dev, Y_dev)
                
                history['loss'].append(loss)
                history['accuracy'].append(train_accuracy)
                history['dev_accuracy'].append(dev_accuracy)
                history['epochs'].append(epoch)
                history['times'].append(time.time() - start_time)
                
                if verbose:
                    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | Dev Acc: {dev_accuracy*100:.2f}% | Time: {epoch_time:.3f}s")
            
            # Learning rate decay
            self.learning_rate = self.learning_rate * (1 / (1 + 0.001 * epoch))
        
        return history
    
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
    
    def plot_comparison(self, histories, gradient_types):
        """Plot comparison of different gradient descent methods"""
        plt.figure(figsize=(18, 5))
        
        # Plot Dev Accuracy
        plt.subplot(1, 3, 1)
        for i, (history, gd_type) in enumerate(zip(histories, gradient_types)):
            plt.plot(history['epochs'], [acc*100 for acc in history['dev_accuracy']], 
                    label=f"{gd_type}", linewidth=2, marker='o', markersize=3)
        plt.xlabel("Epochs")
        plt.ylabel("Development Accuracy (%)")
        plt.title("Development Accuracy Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Loss
        plt.subplot(1, 3, 2)
        for i, (history, gd_type) in enumerate(zip(histories, gradient_types)):
            plt.plot(history['epochs'], history['loss'], 
                    label=f"{gd_type}", linewidth=2, marker='o', markersize=3)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Training Time
        plt.subplot(1, 3, 3)
        for i, (history, gd_type) in enumerate(zip(histories, gradient_types)):
            plt.plot(history['epochs'], history['times'], 
                    label=f"{gd_type}", linewidth=2, marker='o', markersize=3)
        plt.xlabel("Epochs")
        plt.ylabel("Cumulative Time (seconds)")
        plt.title("Training Time Comparison")
        plt.grid(True, alpha=0.3)
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
        

# Comparison of Gradient Descent Types: Batch vs Mini-batch vs Stochastic
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading MNIST data...")
    data = pd.read_csv('digit-recognizer/train.csv').to_numpy()
    np.random.shuffle(data)

    m, n = data.shape
    data_dev = data[:1000].T
    Y_dev = data_dev[0].astype(int)
    X_dev = data_dev[1:] / 255.0

    # Use smaller training set for faster comparison (first 5000 samples)
    data_train = data[1000:6000].T
    Y_train = data_train[0].astype(int)
    X_train = data_train[1:] / 255.0

    print(f"Training samples: {X_train.shape[1]}, Dev samples: {X_dev.shape[1]}")
    print(f"Network architecture: 784-100-100-100-10")
    print("\n" + "="*80)

    # Gradient descent configurations
    gradient_types = ['batch', 'mini-batch', 'stochastic']
    batch_sizes = [X_train.shape[1], 64, 1]  # Full batch, mini-batch, single sample
    colors = ['blue', 'green', 'red']
    
    histories = []
    results = []
    
    # Test each gradient descent type
    for i, (gd_type, batch_size) in enumerate(zip(gradient_types, batch_sizes)):
        print(f"\nTesting {gd_type.upper()} Gradient Descent")
        print(f"Batch size: {batch_size if gd_type != 'batch' else 'Full dataset'}")
        print("-" * 50)
        
        # Create fresh network for each test
        nn = NeuralNetwork(
            layer_sizes=[784, 100, 100, 100, 10],
            activation_functions=['relu', 'relu', 'relu', 'softmax'],
            learning_rate=0.02,  # Reduced from 0.1 to prevent overfitting
            regularization=0.01
        )
        
        # Train with specific gradient descent type
        start_time = time.time()
        history = nn.fit_with_gradient_type(
            X_train, Y_train, X_dev, Y_dev,
            gradient_type=gd_type,
            epochs=50,  # Fixed epochs for fair comparison
            batch_size=batch_size,
            verbose=True
        )
        total_time = time.time() - start_time
        
        # Final evaluation
        final_train_acc = nn.get_accuracy(X_train, Y_train)
        final_dev_acc = nn.get_accuracy(X_dev, Y_dev)
        
        histories.append(history)
        results.append({
            'type': gd_type,
            'batch_size': batch_size,
            'total_time': total_time,
            'final_train_acc': final_train_acc,
            'final_dev_acc': final_dev_acc,
            'final_loss': history['loss'][-1] if history['loss'] else None
        })
        
        print(f"\nFinal Results for {gd_type.upper()}:")
        print(f"  Total training time: {total_time:.2f} seconds")
        print(f"  Final training accuracy: {final_train_acc*100:.2f}%")
        print(f"  Final dev accuracy: {final_dev_acc*100:.2f}%")
        print(f"  Final loss: {history['loss'][-1]:.4f}" if history['loss'] else "N/A")
        print("="*50)

    # Print comparison summary
    print("\\n" + "="*80)
    print("GRADIENT DESCENT COMPARISON SUMMARY (50 Epochs Each)")
    print("="*80)
    print(f"{'Method':<15} {'Batch Size':<12} {'Time (s)':<10} {'Train Acc':<12} {'Dev Acc':<10} {'Final Loss':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['type'].title():<15} {str(result['batch_size']):<12} {result['total_time']:<10.2f} "
              f"{result['final_train_acc']*100:<12.2f} {result['final_dev_acc']*100:<10.2f} {result['final_loss']:<12.4f}")
    
    print("\\nDetailed Analysis:")
    
    # Find best performing method for each metric
    best_time = min(results, key=lambda x: x['total_time'])
    best_dev_acc = max(results, key=lambda x: x['final_dev_acc'])
    best_loss = min(results, key=lambda x: x['final_loss'])
    best_train_acc = max(results, key=lambda x: x['final_train_acc'])
    
    print(f"• Fastest Training: {best_time['type'].title()} ({best_time['total_time']:.2f}s)")
    print(f"• Best Dev Accuracy: {best_dev_acc['type'].title()} ({best_dev_acc['final_dev_acc']*100:.2f}%)")
    print(f"• Lowest Loss: {best_loss['type'].title()} ({best_loss['final_loss']:.4f})")
    print(f"• Best Train Accuracy: {best_train_acc['type'].title()} ({best_train_acc['final_train_acc']*100:.2f}%)")
    
    # Calculate efficiency metrics
    print("\\nEfficiency Analysis:")
    for result in results:
        time_per_epoch = result['total_time'] / 50
        acc_per_time = result['final_dev_acc'] / result['total_time']
        print(f"• {result['type'].title()}: {time_per_epoch:.3f}s/epoch, {acc_per_time:.4f} acc/sec")
    
    print("\\nCharacteristics Observed:")
    print("• Batch GD: Most stable convergence, uses entire dataset per update")
    print("• Mini-batch GD: Good balance of speed and stability, vectorized efficiency") 
    print("• Stochastic GD: Fastest updates but noisiest convergence pattern")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    nn_plot = NeuralNetwork([784, 100, 100, 100, 10])  # Dummy network for plotting
    nn_plot.plot_comparison(histories, gradient_types)
    
    print("\nComparison completed!")