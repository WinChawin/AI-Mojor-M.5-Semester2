import torch
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(2).repeat(40)
r = np.random.normal(z+1, 0.25)
t = np.random.uniform(0, np.pi, 80)
x = r * np.cos(t)
y = r * np.sin(t)
X = np.array([x, y]).T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# linear_layer = torch.nn.Linear(2, 3).to(device)
# print("Linear layer weights:", linear_layer.weight)
# print("Linear Layer bias:", linear_layer.bias)

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.lyer1 = torch.nn.Linear(2, 32)  
        self.lyer2 = torch.nn.Linear(32, 32)
        self.lyer3 = torch.nn.Linear(32, 2)

        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        h1 = self.lyer1(x)
        a1 = torch.nn.functional.relu(h1)
        a1 = self.dropout1(a1)
        # a1 = torch.nn.functional.relu(h1)
        h2 = self.lyer2(a1)
        a2 = torch.nn.functional.relu(h2)
        a2 = self.dropout2(a2)
        # a2 = torch.nn.functional.relu(h2)
        out = self.lyer3(a2)
        return out

X = torch.tensor(X, dtype=torch.float32).to(device)
z = torch.tensor(z, dtype=torch.long).to(device)
model = SimpleNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
cross_entropy = torch.nn.CrossEntropyLoss()

loss_list = []

best = float('inf')
patience = 10
bad = 0

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X)
    loss = cross_entropy(outputs, z)
    loss.backward()
    optimizer.step()
    val_loss = loss.item()
    loss_list.append(val_loss)

    if(val_loss < best):
        best = val_loss
        bad = 0
        print(f'Saving model at epoch {epoch+1} with loss {best:.4f}')
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        bad += 1
        if bad >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

model.eval()  
with torch.no_grad():
    out = model(X)
    preds = torch.argmax(out, dim=1)
    acc = (preds == z).float().mean().item()
    print(f'Accuracy: {acc*100:.2f}%')

X_cpu = X.cpu()
z_cpu = z.cpu()
preds_cpu = preds.cpu().numpy()

margin = 0.5
xx, yy = np.meshgrid(
    np.arange(X_cpu[:, 0].min()-margin, X_cpu[:, 0].max()+margin, 0.01),
    np.arange(X_cpu[:, 1].min()-margin, X_cpu[:, 1].max()+margin, 0.01)
)
gr = np.c_[xx.ravel(), yy.ravel()]
g_t = torch.from_numpy(gr).float().to(device)
with torch.no_grad():
    out = model(g_t)
    grid_preds = out.argmax(dim=1).cpu().numpy()
ZG = grid_preds.reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, ZG, cmap='bwr', alpha=0.2)
plt.scatter(X_cpu[:, 0], X_cpu[:, 1], c=z_cpu, cmap='bwr', s=40, edgecolors='k')

mis = (preds_cpu != z_cpu.numpy())
if mis.any():
    plt.scatter(X_cpu[mis, 0], X_cpu[mis, 1], facecolors='none', edgecolors='k', linewidths=1.5, s=120, marker='o')

plt.title(f'Data and Decision Boundary (accuracy={acc*100:.2f}%)')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(alpha=0.3, linestyle='--')
plt.savefig('nn_decision_boundary.png', dpi=200, bbox_inches='tight')

plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(alpha=0.3, linestyle='--')
plt.savefig('nn_training_loss.png', dpi=200, bbox_inches='tight')

plt.show()