from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

criterion = nn.BCEWithLogitsLoss()


class ModularNN(nn.Module):

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 hidden_layers: List[int],
                 activation=nn.ReLU(),
                 dropout_rate: float = .5):
        """
        Flexible neural network with customizable architecture.

        :param inp_features: Number of input features. Eg: X_train.shape[1]
        :param out_features: Number of output features.
        :param hidden_layers: List of hidden layers where len(hidden_layers) is the number of layers and
        the actual number is the amount of neurons in the hidden layer. Eg: [64, 128]
        :param activation: Activation function like ReLU or Sigmoid.
        :param dropout_rate: Rate of dropout.
        """
        super(ModularNN, self).__init__()
        layers = []
        previous_layer_size = inp_features
        for layer_size in hidden_layers:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            previous_layer_size = layer_size
        layers.append(nn.Linear(previous_layer_size, out_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def model_train(model, X_train, y_train, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for i in range(epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {i}, Loss: {loss.item()}")


def model_eval(model, X_test, y_test):
    with torch.no_grad():
        y_pred = model(X_test)
        loss = criterion(y_pred, y_test)
        print(f"Loss: {loss.item()}")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    samples = 1000
    features = 10
    hidden_layers = [64, 128]
    output_layers = 1

    X = np.random.rand(samples, features).astype(np.float32)
    y = (np.sum(X, axis=1) > 5).astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = ModularNN(
        inp_features=X_train.shape[1],
        out_features=output_layers,
        hidden_layers=hidden_layers,
        activation=nn.ReLU(),
        dropout_rate=0.5
    )
    print(f"Number of trainable parameters: {model.count_trainable_params()}")

    model_train(
        model,
        X_train,
        y_train,
        epochs=100,
        learning_rate=0.01
    )

    model_eval(model, X_test, y_test)
