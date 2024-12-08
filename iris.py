import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report

from churn_data_preprocessing import X_train, y_train, X_test, y_test


class IrisClassification(nn.Module):

    def __init__(self, max_features):
        super(IrisClassification, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(max_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_model(X_train_features, y_train_features, learning_rate, epochs, inp_features):
    X_train_tensor = torch.tensor(X_train_features, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_features, dtype=torch.float32).view(-1, 1)

    iris_model = IrisClassification(inp_features)
    optimizer = torch.optim.Adam(iris_model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):
        iris_model.train()
        y_pred = iris_model(X_train_tensor)
        loss = loss_function(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}, at epoch {epoch}")
    return iris_model


def evaluate_model(X_features, y_features, nn_model):
    X_test_tensor = torch.tensor(X_features, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_features, dtype=torch.float32)

    nn_model.eval()

    with torch.no_grad():
        y_pred_prob = torch.sigmoid(nn_model(X_test_tensor))
        y_pred = (y_pred_prob > 0.5).float()

        accuracy = (y_pred.squeeze() == y_test_tensor).float().mean()
        print(f"Accuracy: {round(accuracy.item() * 100, 2)}%")

        y_pred_np = y_pred.numpy()
        y_test_np = y_test_tensor.numpy()

        print("\nClassification Report")
        print(classification_report(y_test_np, y_pred_np))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test_np, y_pred_np))

    return y_pred, y_pred_prob


if __name__ == '__main__':
    model = train_model(
        inp_features=X_train.shape[1],
        X_train_features=X_train,
        y_train_features=y_train,
        learning_rate=0.01,
        epochs=200,
    )
    y_hat, y_hat_probability = evaluate_model(
        X_features=X_test,
        y_features=y_test,
        nn_model=model
    )

