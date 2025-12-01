import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
 
candidates = [
    r"D:\SEMESTERS 2\digit-recognizer\train.csv",  
    r"D:\digit-recognizer\train.csv",
    os.path.join(os.path.dirname(__file__), "digit-recognizer", "train.csv"),
    os.path.join(os.path.dirname(__file__), "digit-recognizer", "train_clean.csv"),
    os.path.join("digit-recognizer", "train.csv"),
    os.path.join("digit-recognizer", "train_clean.csv"),
]
data = None
for p in candidates:
    if os.path.exists(p):
        data = pd.read_csv(p)
        break
if data is None:
    raise FileNotFoundError(f"ไม่พบไฟล์ train.csv ในพาธที่คาด: {candidates}")
 
Y = np.array(data['label'])
X = np.array(data.drop('label', axis=1)).T
X = X / 255.0  
 
def relu(Z):
    return np.maximum(0, Z)
 
def deriv_relu(Z):
    return Z > 0
 
def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)
 
def one_hot(Y):
    m = Y.size
    one_hot_Y = np.zeros((10, m))
    one_hot_Y[Y, np.arange(m)] = 1
    return one_hot_Y
 
def get_predict(A2):
    return np.argmax(A2, axis=0)
 
def get_accuracy(A2, Y):
    pred = get_predict(A2)
    return np.mean(pred == Y)
 
class TrainNN:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # ปรับ scaling factor สำหรับ ReLU initialization
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))
 
    def forward_prop(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = relu(Z1)  # Changed from sigmoid to relu
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
 
    def back_prop(self, Z1, A1, Z2, A2, X, Y, alpha):
        m = X.shape[1]
        one_hot_Y = one_hot(Y)
 
        dZ2 = A2 - one_hot_Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
 
        dZ1 = np.dot(self.W2.T, dZ2) * deriv_relu(Z1)  # Changed from deriv_sigmoid to deriv_relu
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
 
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
 
    def fit(self, X, Y, alpha=0.05, n_iters=1000):
        total_time = 0
        times = []
       
        for i in range(1, n_iters + 1):
            start_time = time.time()
           
            Z1, A1, Z2, A2 = self.forward_prop(X)
            self.back_prop(Z1, A1, Z2, A2, X, Y, alpha)
           
            epoch_time = time.time() - start_time
            times.append(epoch_time)
            total_time += epoch_time
 
            if i % 100 == 0 or i == n_iters:
                acc = get_accuracy(A2, Y)
                avg_time = total_time / i
                print(f"Iteration {i}/{n_iters} - Accuracy: {acc*100:.2f}% - Time: {epoch_time:.4f}s - Avg time/epoch: {avg_time:.4f}s")
       
        # สรุปเวลาหลังเทรนเสร็จ
        print("\nTraining Summary:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per epoch: {np.mean(times):.4f}s")
        print(f"Min epoch time: {min(times):.4f}s")
        print(f"Max epoch time: {max(times):.4f}s")
 
def show_data_grid(data_df, true_labels, model=None, rows=10, cols=10, start_idx=0,
                   figsize_per_cell=(1.2, 1.2), dpi=90):
    """
    แสดงกริดของภาพจาก DataFrame ที่มีคอลัมน์พิกเซล (มีคอลัมน์ 'label' อยู่แล้ว)
    - true_labels: array-like ของ label จริง
    - model: ถ้ามี จะใช้ model.forward_prop บน X_normalized เพื่อได้ prediction
    """
    pixel_df = data_df.drop('label', axis=1)
    n_cells = rows * cols
    indices = list(range(start_idx, min(start_idx + n_cells, pixel_df.shape[0])))
 
    fig_w = cols * figsize_per_cell[0]
    fig_h = rows * figsize_per_cell[1]
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = axes.flatten()
 
    preds = None
    if model is not None:
        X_all = np.array(pixel_df).T.astype(float) / 255.0
        _, _, _, A2_all = model.forward_prop(X_all)
        preds = np.argmax(A2_all, axis=0)
 
    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(indices):
            ax.axis('off')
            continue
        idx = indices[ax_idx]
        img = pixel_df.iloc[idx].values.astype(float).reshape(28, 28)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        true_lab = int(true_labels[idx])
        title_txt = f"{true_lab}"
        if preds is not None:
            pred_lab = int(preds[idx])
            color = 'green' if pred_lab == true_lab else 'red'
            ax.text(0.01, 0.08, str(true_lab), color='lime', fontsize=8, transform=ax.transAxes,
                    bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'))
            ax.text(0.01, 0.88, str(pred_lab), color=color, fontsize=8, transform=ax.transAxes,
                    bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'))
        else:
            ax.set_title(title_txt, fontsize=8, color='lime')
        ax.axis('off')
 
    plt.tight_layout()
    plt.show()
 
model = TrainNN()
model.fit(X, Y, alpha=1, n_iters=1500)
 
show_data_grid(data, np.array(data['label']), model=model, rows=10, cols=10, start_idx=0,
               figsize_per_cell=(1.0, 1.0), dpi=90)