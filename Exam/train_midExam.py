import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('Exam\\cleaned_dataset2.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

feature_columns = [str(i) for i in range(784)]
X = df[feature_columns].to_numpy().astype(np.int64)
Y = df['Label'].to_numpy().astype(np.int64)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, num_classes)


    def forward(self, x): 
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
model = NeuralNet(input_size=784, hidden_size=128, num_classes=10).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 130

loss_list = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

print("Training complete.")

model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

plt.plot(range(1, epochs + 1), loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()


plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    image = X_test[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    with torch.no_grad():
        input_tensor = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        _, predicted_label = torch.max(output.data, 1)

    if predicted_label.item() == Y_test[i]: 
        plt.title(f'Predicted: {predicted_label.item()}, Actual: {Y_test[i]}', color='green')
    else:
        plt.title(f'Predicted: {predicted_label.item()}, Actual: {Y_test[i]}', color='red')
    plt.axis('off')
plt.show()