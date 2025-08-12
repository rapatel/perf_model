"""
Pytorch cost model for predicting performance based on operation, system configuration, and workload characteristics.
"""

import torch
import logging


class CostModel(torch.nn.Module):
    def __init__(self, num_layers: int, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )
        self.output_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Predicts the cost based on the input data.
        :param input_data: Input tensor containing operation and system configuration data.
        :return: Predicted cost as a tensor.
        """
        x = input_data
        for layer in self.hidden_layers:
            x = torch.nn.ReLU()(layer(x))
        return self.output_layer(x)

    def fit(
        self,
        training_data: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        """
        Trains the cost model using the provided training data and targets.
        :param training_data: Input tensor for training.
        :param targets: Target tensor for training.
        :param epochs: Number of epochs to train.
        :param learning_rate: Learning rate for the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(training_data)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        return self
