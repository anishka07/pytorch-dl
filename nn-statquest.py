import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicNN(nn.Module):

    def __init__(self):
        super(BasicNN, self).__init__()

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(1.7), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(-.85), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(1.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-.85), requires_grad=False)

    def forward(self, inp_param):
        inp_to_top_relu = inp_param * self.w00 + self.b00
        top_relu_op = F.relu(inp_to_top_relu)
        scaled_top_relu = top_relu_op * self.w01

        inp_to_bottom_relu = inp_param * self.w10 + self.b10
        bottom_relu_op = F.relu(inp_to_bottom_relu)
        scaled_bottom_relu = bottom_relu_op * self.w11

        inp_to_final_relu = scaled_top_relu + scaled_bottom_relu + self.final_bias
        output = F.relu(inp_to_final_relu)
        return output


if __name__ == '__main__':
    input_doses = torch.linspace(0, 1, 11)
    model = BasicNN()
    output_values = model(input_doses)

    plt.figure(figsize=(12, 8))
    sns.set(style='whitegrid')
    sns.lineplot(x=input_doses, y=output_values, color='green', linewidth=2.5)
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
    plt.show()
