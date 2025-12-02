import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def char_to_num(labels):
    """
    แปลงตัวอักษร A-Z เป็นตัวเลข 0-25
    ถ้า labels เป็นตัวเลขอยู่แล้ว จะ return คืนโดยไม่เปลี่ยนแปลง
    """
    if isinstance(labels, (list, np.ndarray)):
        if len(labels) > 0 and isinstance(labels[0], str):
            # แปลงตัวอักษรเป็นตัวเลข
            return np.array([ord(label.upper()) - ord('A') for label in labels])
    return np.array(labels, dtype=int)

def num_to_char(numbers):
    """
    แปลงตัวเลข 0-25 กลับเป็นตัวอักษร A-Z
    """
    return [chr(num + ord('A')) for num in numbers]

def create_label_mapping(labels):
    """
    สร้าง mapping ระหว่าง original labels กับตัวเลข
    คืนค่า (label_to_num_dict, num_to_label_dict)
    """
    unique_labels = sorted(list(set(labels)))
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    num_to_label = {i: label for i, label in enumerate(unique_labels)}
    return label_to_num, num_to_label

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions=None,
                 learning_rate=0.1, regularization=0.01):
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # ถ้าไม่กำหนด activation_function: hidden = relu, output = softmax
        if activation_functions is None:
            self.activation_functions = ['relu'] * (self.num_layers - 2) + ['softmax']
        else:
            if len(activation_functions) != self.num_layers - 1:
                raise ValueError("len(activation_functions) ต้องเท่ากับ num_layers - 1")
            self.activation_functions = activation_functions
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        self._init_params()
        
        # training_history จะถูกเติมหลังจาก fit_with_gradient_type() หรือ train_until_accuracy() รัน
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'dev_accuracy': [],
            'epochs': [],
            'times': []
        }
     
    def _init_params(self):
        """Initialize weights and biases (He initialization สำหรับ ReLU)"""
        for i in range(1, self.num_layers):
            fan_in = self.layer_sizes[i-1]
            self.weights[f'W{i}'] = np.random.randn(self.layer_sizes[i], fan_in) * np.sqrt(2.0 / fan_in)
            self.biases[f'b{i}'] = np.zeros((self.layer_sizes[i], 1))
    
    # Activation functions
    def _relu(self, z):
        return np.maximum(z, 0)
    
    def _relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def _tanh(self, z):
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        return 1.0 - np.tanh(z)**2
    
    def _sigmoid(self, z):
        z_clipped = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1.0 - s)
    
    def _softmax(self, z):
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
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
        elif activation == 'linear':
            return z
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _apply_activation_derivative(self, z, activation):
        """Apply อนุพันธ์ของ activation ตามชื่อ"""
        if activation == 'relu':
            return self._relu_derivative(z)
        elif activation == 'tanh':
            return self._tanh_derivative(z)
        elif activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif activation == 'linear':
            return np.ones_like(z)
        else:
            # ปกติจะไม่ใช้ derivative ของ softmax ตรง ๆ (ใช้คู่กับ cross-entropy แล้วรวม)
            raise ValueError(f"Unsupported activation derivative for: {activation}")
    
    # Utils for labels
    def _one_hot_encode(self, Y):
        """
        แปลง label เป็น one-hot
        Y: vector shape (m,) ของ integer class (0..C-1)
        """
        num_classes = self.layer_sizes[-1]
        one_hot_Y = np.zeros((num_classes, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y
    
    # Forward pass
    def forward_propagation(self, X):
        """
        X: input data shape (n_features, m)
        return: activations dict, z_values dict
        """
        activations = {'A0': X}
        z_values = {}
        
        for i in range(1, self.num_layers):
            Wi = self.weights[f'W{i}']
            bi = self.biases[f'b{i}']
            Ai_1 = activations[f'A{i-1}']
            
            Z = np.dot(Wi, Ai_1) + bi
            z_values[f'Z{i}'] = Z
            
            A = self._apply_activation(Z, self.activation_functions[i-1])
            activations[f'A{i}'] = A
        
        return activations, z_values
    
    # Loss function
    def compute_loss(self, Y_pred, Y_true):
        """
        คำนวณ cross-entropy loss + L2 regularization
        Y_pred: softmax output, shape (num_classes, m)
        Y_true: vector of true labels, shape (m,)
        """
        m = Y_true.size
        
        # Cross-entropy loss
        probs_correct = Y_pred[Y_true, np.arange(m)]
        log_probs = -np.log(probs_correct + 1e-8)
        data_loss = np.sum(log_probs) / m
        
        # L2 regularization
        reg_loss = 0.0
        for i in range(1, self.num_layers):
            reg_loss += np.sum(self.weights[f'W{i}']**2)
        reg_loss = (self.regularization / (2.0 * m)) * reg_loss
        
        return data_loss + reg_loss
    
    # Backward pass
    def backward_propagation(self, activations, z_values, X, Y):
        """
        ทำ Backpropagation ทั้ง network
        X: (n_features, m)
        Y: vector of labels (m,)
        """
        m = Y.size
        gradients = {}
        one_hot_Y = self._one_hot_encode(Y)
        
        # เริ่มจาก output layer: softmax + cross-entropy → dZ = A_L - Y_one_hot
        dZ = activations[f'A{self.num_layers-1}'] - one_hot_Y
        
        # ไล่ย้อนจาก layer L, L-1, ..., 1
        for i in range(self.num_layers-1, 0, -1):
            Ai_1 = activations[f'A{i-1}']
            Wi = self.weights[f'W{i}']
            
            gradients[f'dW{i}'] = (1.0/m) * np.dot(dZ, Ai_1.T) + (self.regularization/m) * Wi
            gradients[f'db{i}'] = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)
            
            if i > 1:
                dA_prev = np.dot(Wi.T, dZ)
                Zi_1 = z_values[f'Z{i-1}']
                dZ = dA_prev * self._apply_activation_derivative(Zi_1, self.activation_functions[i-2])
        
        return gradients
    
    # Parameter update
    def update_parameters(self, gradients):
        """Update weights and biases using gradients"""
        for i in range(1, self.num_layers):
            self.weights[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']
            self.biases[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']
    
    # Prediction & Accuracy
    def predict(self, X):
        """คืนค่า class ที่ทำนายได้ (argmax)"""
        activations, _ = self.forward_propagation(X)
        AL = activations[f'A{self.num_layers-1}']
        return np.argmax(AL, axis=0)
    
    def predict_with_mapping(self, X, num_to_label_dict):
        """ทำนายและแปลงกลับเป็น original labels"""
        predictions = self.predict(X)
        if num_to_label_dict:
            return [num_to_label_dict[pred] for pred in predictions]
        return predictions
    
    def get_accuracy(self, X, Y):
        """คำนวณ accuracy = (จำนวนทายถูก / ทั้งหมด)"""
        preds = self.predict(X)
        return np.sum(preds == Y) / Y.size
    
    # Training with different gradient types (by epochs)
    def fit_with_gradient_type(self, X_train, Y_train,
                               X_dev, Y_dev,
                               gradient_type='mini-batch',
                               epochs=100, batch_size=64,
                               verbose=True, log_interval=5):
        """
        Train network ด้วย gradient descent หลายแบบ:
          - 'batch'      : ใช้ทั้งชุดข้อมูลต่อ 1 update
          - 'stochastic' : อัปเดตทีละตัวอย่าง (SGD)
          - 'mini-batch' : แบ่ง batch ย่อย ๆ (default)
        
        X_train: (n_features, m_train)
        Y_train: (m_train,)
        X_dev, Y_dev: dev/validation set
        """
        m = X_train.shape[1]
        
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
                # Batch GD
                activations, z_values = self.forward_propagation(X_train)
                gradients = self.backward_propagation(activations, z_values, X_train, Y_train)
                self.update_parameters(gradients)
            
            elif gradient_type == 'stochastic':
                # Stochastic GD
                indices = np.random.permutation(m)
                for i in indices:
                    X_sample = X_train[:, i:i+1]      # (n_features, 1)
                    Y_sample = Y_train[i:i+1]         # (1,)
                    activations, z_values = self.forward_propagation(X_sample)
                    gradients = self.backward_propagation(activations, z_values, X_sample, Y_sample)
                    self.update_parameters(gradients)
            
            else:
                # Mini-batch GD (default)
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
            
            # log ทุก ๆ log_interval epochs (default = 5)
            if epoch % log_interval == 0:
                activations, _ = self.forward_propagation(X_train)
                AL = activations[f'A{self.num_layers-1}']
                loss = self.compute_loss(AL, Y_train)
                train_acc = self.get_accuracy(X_train, Y_train)
                dev_acc = self.get_accuracy(X_dev, Y_dev)
                
                history['loss'].append(loss)
                history['accuracy'].append(train_acc)
                history['dev_accuracy'].append(dev_acc)
                history['epochs'].append(epoch)
                history['times'].append(time.time() - start_time)
                
                if verbose:
                    print(f"[FixedEpoch] Epoch {epoch:3d} | Loss: {loss:.4f} | "
                          f"Train Acc: {train_acc*100:.2f}% | Dev Acc: {dev_acc*100:.2f}% | "
                          f"Epoch Time: {epoch_time:.3f}s")
            
            # Learning rate decay (แบบง่าย)
            self.learning_rate = self.learning_rate * (1.0 / (1.0 + 0.001 * epoch))
        
        total_time = time.time() - start_time
        print(f"\n[FixedEpoch] Total training time: {total_time:.2f} seconds")
        
        # เซฟเข้า object เพื่อให้ plot ทีหลังได้
        self.training_history = history
        return history
    
    # Plot history
    def plot_training_history(self):
        """Plot loss และ accuracy จาก training_history ที่ได้จาก fit_with_gradient_type หรือ train_until_accuracy"""
        if len(self.training_history['epochs']) == 0:
            print("No training history found. Run training first.")
            return
        
        epochs = self.training_history['epochs']
        loss = self.training_history['loss']
        acc = [a * 100 for a in self.training_history['accuracy']]
        dev_acc = [a * 100 for a in self.training_history['dev_accuracy']]
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label="Train Accuracy")
        plt.plot(epochs, dev_acc, label="Dev Accuracy", linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy over Epochs")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label="Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print network summary"""
        print("Neural Network Summary:")
        print("=" * 50)
        print(f"Number of layers (not counting input): {self.num_layers - 1}")
        print(f"Layer sizes: {self.layer_sizes}")
        print(f"Activation functions: {self.activation_functions}")
        print(f"Learning rate (current): {self.learning_rate}")
        print(f"Regularization (L2 lambda): {self.regularization}")
        print("=" * 50)


# แบบที่ 1: เทรนจนถึง accuracy เป้าหมาย
def train_until_accuracy(model, X_train, Y_train,
                         X_dev, Y_dev,
                         target_accuracy=0.90,
                         max_epochs=100,
                         batch_size=64,
                         verbose=True):
    """
    เทรนโมเดลด้วย mini-batch จนกว่า dev accuracy จะ >= target_accuracy
    หรือจนถึง max_epochs (อย่างใดอย่างหนึ่งมาถึงก่อน)
    """
    m = X_train.shape[1]
    
    history = {
        'loss': [],
        'accuracy': [],
        'dev_accuracy': [],
        'epochs': [],
        'times': []
    }
    
    start_time = time.time()
    
    for epoch in range(max_epochs + 1):
        epoch_start = time.time()
        
        # Mini-batch
        indices = np.random.permutation(m)
        X_shuffled = X_train[:, indices]
        Y_shuffled = Y_train[indices]
        
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]
            activations, z_values = model.forward_propagation(X_batch)
            gradients = model.backward_propagation(activations, z_values, X_batch, Y_batch)
            model.update_parameters(gradients)
        
        epoch_time = time.time() - epoch_start
        
        # คำนวณ metrics
        activations, _ = model.forward_propagation(X_train)
        AL = activations[f'A{model.num_layers-1}']
        loss = model.compute_loss(AL, Y_train)
        train_acc = model.get_accuracy(X_train, Y_train)
        dev_acc = model.get_accuracy(X_dev, Y_dev)
        
        history['loss'].append(loss)
        history['accuracy'].append(train_acc)
        history['dev_accuracy'].append(dev_acc)
        history['epochs'].append(epoch)
        history['times'].append(time.time() - start_time)
        
        if verbose:
            print(f"[UntilAcc] Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | Dev Acc: {dev_acc*100:.2f}% | "
                  f"Epoch Time: {epoch_time:.3f}s")
        
        # LR decay เหมือนกัน
        model.learning_rate = model.learning_rate * (1.0 / (1.0 + 0.001 * epoch))
        
        # หยุดเมื่อ dev accuracy ถึงเป้า
        if dev_acc >= target_accuracy:
            total_time = time.time() - start_time
            print(f"\nReached target dev accuracy {target_accuracy*100:.2f}% at epoch {epoch}")
            print(f"[UntilAcc] Total training time: {total_time:.2f} seconds")
            break
    
    model.training_history = history
    return history


if __name__ == "__main__":
    # 1. Load + shuffle + split 
    data_path = "data.csv"  
    
    try:
        # อ่านเป็น DataFrame ก่อนเพื่อตรวจสอบข้อมูล
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # สมมติว่า column แรกเป็น label
        labels = df.iloc[:, 0].values  # column แรก
        features = df.iloc[:, 1:].values  # column ที่เหลือ
        
        print(f"Sample labels: {labels[:10]}")
        print(f"Label type: {type(labels[0])}")
        
        # สร้าง label mapping และแปลง labels เป็นตัวเลข
        label_to_num, num_to_label = create_label_mapping(labels)
        print(f"Label mapping: {label_to_num}")
        
        # แปลง labels เป็นตัวเลข
        numeric_labels = np.array([label_to_num[label] for label in labels])
        
        # รวมข้อมูลและ shuffle
        combined_data = np.column_stack([numeric_labels, features])
        np.random.shuffle(combined_data)
        
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found. Please check the file path.")
        exit(1)
    except Exception as e:
        print(f"Error reading data: {e}")
        exit(1)
    
    m, n = combined_data.shape  # m = จำนวนตัวอย่าง, n = 1(label) + features
    
    # dev/test = 1000 ตัวแรก
    data_dev = combined_data[:1000].T
    Y_dev = data_dev[0].astype(int)
    X_dev = data_dev[1:] / 255.0   # normalize
    
    # train = ที่เหลือ
    data_train = combined_data[1000:].T
    Y_train = data_train[0].astype(int)
    X_train = data_train[1:] / 255.0
    
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_dev shape:", X_dev.shape)
    print("Y_dev shape:", Y_dev.shape)
    print("Numeric labels range:", f"{Y_train.min()}-{Y_train.max()}")
    
    # ----- 2. สร้างโมเดล -----
    input_size = X_train.shape[0]           
    num_classes = len(np.unique(Y_train))   # จำนวน unique labels
    
    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Class labels: {sorted(num_to_label.values())}")
    
    layer_sizes = [input_size, 100, 100, num_classes]
    nn = NeuralNetwork(layer_sizes=layer_sizes,
                       activation_functions=['relu', 'relu', 'softmax'],  # กำหนด activation สำหรับแต่ละ layer
                       learning_rate=0.1,
                       regularization=0.01)
    
    nn.summary()
    
    # MODE 1: เทรนจนถึง accuracy ที่กำหนด
    # target_acc = 0.95  # 95%
    # history_until = train_until_accuracy(
    #     nn,
    #     X_train, Y_train,
    #     X_dev,   Y_dev,
    #     target_accuracy=target_acc,
    #     max_epochs=100,
    #     batch_size=64,
    #     verbose=True
    # )
    
    # print("Dev accuracy (final): {:.2f}%".format(history_until['dev_accuracy'][-1] * 100))
    # nn.plot_training_history()
    
    # # ในกรณีนี้เราไม่มี test set แยกต่างหาก (เพราะ Kaggle แยกเป็น test.csv ไม่มี label)
    # # ถ้าอยากวัดเพิ่ม อาจใช้ X_dev,Y_dev เป็นตัวแทน test ชั่วคราว:
    # test_acc = nn.get_accuracy(X_dev, Y_dev)
    # print("Dev/Test accuracy (using dev as test): {:.2f}%".format(test_acc * 100))
    
    # MODE 2: เทรนตามจำนวน epoch ที่กำหนด
    # ถ้าอยากใช้โหมด Fixed Epoch ให้ comment MODE 1 ด้านบน แล้ว uncomment ด้านล่าง:
    
    history_fixed = nn.fit_with_gradient_type(
        X_train, Y_train,
        X_dev,   Y_dev,
        gradient_type='mini-batch',
        epochs=100,
        batch_size=64,
        verbose=True,
        log_interval=5
    )
    
    nn.plot_training_history()
    test_acc = nn.get_accuracy(X_dev, Y_dev)
    print("Dev/Test accuracy (using dev as test): {:.2f}%".format(test_acc * 100))
    
    # ทดสอบการทำนายด้วย sample จาก dev set
    print("\n=== Sample Predictions ===")
    sample_indices = np.random.choice(len(Y_dev), 5, replace=False)
    sample_X = X_dev[:, sample_indices]
    sample_Y_true = Y_dev[sample_indices]
    sample_predictions = nn.predict(sample_X)
    
    for i in range(5):
        true_label = num_to_label[sample_Y_true[i]]
        pred_label = num_to_label[sample_predictions[i]]
        print(f"Sample {i+1}: True = {true_label}, Predicted = {pred_label}")
