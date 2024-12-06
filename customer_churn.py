import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

from churn_data_preprocessing import X_train, X_test, y_train, y_test


class CustomerChurnPredictor(nn.Module):

    def __init__(self, num_features):
        super(CustomerChurnPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# training function
def train_model(train_features, y_train_features, inp_features, epochs, learning_rate):
    """
        Trains the model.
    """
    X_train_tensor = torch.tensor(train_features, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_features, dtype=torch.float32).view(-1, 1)

    churn_model = CustomerChurnPredictor(inp_features)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(churn_model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        # training mode
        churn_model.train()
        y_pred = churn_model(X_train_tensor)  # forward propagation
        loss = loss_function(y_pred, y_train_tensor)  # calculate loss
        # backward pass and optimize
        '''
        if coding manually: (not optimal for deep neural networks)
        with torch.no_grad():
            model.linear.weight -= learning_rate * model.linear.weight.grad
            model.linear.bias -= learning_rate * model.linear.bias.grad
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

    return churn_model


# evaluation function
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
        train_features=X_train,
        y_train_features=y_train,
        inp_features=X_train.shape[1],
        epochs=1000,
        learning_rate=0.1
    )
    evaluate_model(X_features=X_test, y_features=y_test, nn_model=model)
