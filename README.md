# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model

![image](https://github.com/user-attachments/assets/badae029-1f7c-4992-b7a9-5b265442bd90)


## DESIGN STEPS
### STEP 1: 
Load the Iris dataset using a suitable library.

### STEP 2: 
Preprocess the data by handling missing values and normalizing features.


### STEP 3: 
Split the dataset into training and testing sets.


### STEP 4: 
Train a classification model using the training data.


### STEP 5: 
Evaluate the model on the test data and calculate accuracy


### STEP 6: 
Display the test accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: Renusri Naraharashetty

### Register Number: 212223240139

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
     

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
torch.manual_seed(32)
model = Model()
     

df = pd.read_csv('/content/iris.csv')
df.head()

X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
     

trainloader = DataLoader(X_train, batch_size=60, shuffle=True)

testloader = DataLoader(X_test, batch_size=60, shuffle=False)
     


torch.manual_seed(4)
model = Model()
     

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
     

trainloader = DataLoader(X_train, batch_size=60, shuffle=True)

testloader = DataLoader(X_test, batch_size=60, shuffle=False)
     


torch.manual_seed(4)
model = Model()
     

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

plt.plot(range(epochs), [loss.detach().numpy() for loss in losses])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()

plt.plot(range(epochs), [loss.detach().numpy() for loss in losses])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()

torch.save(model.state_dict(), 'IrisDatasetModel.pt')
     

new_model = Model()
new_model.load_state_dict(torch.load('IrisDatasetModel.pt'))
new_model.eval()
     
Model(
  (fc1): Linear(in_features=4, out_features=8, bias=True)
  (fc2): Linear(in_features=8, out_features=9, bias=True)
  (out): Linear(in_features=9, out_features=3, bias=True)
)

with torch.no_grad():
    y_val = new_model.forward(X_test)
    loss = criterion(y_val, y_test)
print(f'{loss:.8f}')

mystery_iris = torch.tensor([5.6,3.7,2.2,0.5])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor','Mystery iris']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

    ax.scatter(mystery_iris[plots[i][0]],mystery_iris[plots[i][1]], color='y')

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
plt.show()
     
with torch.no_grad():
    print(new_model(mystery_iris))
    print()
    print(labels[new_model(mystery_iris).argmax()])
```

### Dataset Information

![image](https://github.com/user-attachments/assets/0ad92715-6735-4c1e-b13b-b47888b37d37)


### OUTPUT

![image](https://github.com/user-attachments/assets/59fe0803-22f9-41bd-bce8-c7cb956a8932)

## Confusion Matrix

![image](https://github.com/user-attachments/assets/034590de-d9a4-493f-b681-1f5b330eb309)


## Classification Report

![image](https://github.com/user-attachments/assets/155e0324-5a39-405c-8a0c-0976b74f2c1c)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/43eb99fa-aeed-40e5-b005-782d925160b4)


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch.
