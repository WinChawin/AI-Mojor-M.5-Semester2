import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('digit-recognizer/train.csv')
print(dataset.shape)
print(dataset.head())

# Separate input values form labels
y_train = dataset['label']
x_train = dataset.drop('label', axis=1)

image = x_train.iloc[100].values
image = image.reshape(28, 28)

plt.imshow(image, cmap='gray')
plt.title(f'Label: {y_train.iloc[8]}')
plt.show() 