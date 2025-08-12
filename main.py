"""
Performance Model Application
This script serves as the entry point for the performance model application.
"""

import argparse
import logging
import torch
import os

import pytorch.cost_model as pt_cost_model
from roofline.roofline_model import RooflineModel
from utils import generate_training_data


# Global variables for file paths
WEIGHT_FP_ = os.path.join("data", "cost_model_weights.pth")
NORMALIZATION_FP_ = os.path.join("data", "normalization.pt")


def train_model(num_samples: int = 0) -> None:
    """
    Function to train the cost model.
    This function generates training data and initializes the cost model.
    """
    logging.info("Generating training data...")
    input_data, targets, input_max, target_max = generate_training_data(num_samples)
    logging.info(f"Generated {len(input_data)} training samples.")
    logging.info(
        "Normalized values for input and target tensors: %s, %s.", input_max, target_max
    )

    # Initialize the cost model
    model = pt_cost_model.CostModel(num_layers=3, input_size=input_data.shape[1])
    logging.info("Cost model initialized.")

    # Train the model
    model.fit(input_data, targets, epochs=25, learning_rate=0.001)
    logging.info("Model training completed.")

    # Save the trained model into ./data/ directory
    logging.info("Saving model weights to %s", WEIGHT_FP_)
    torch.save(model.state_dict(), WEIGHT_FP_)
    torch.save({"input_max": input_max, "target_max": target_max}, NORMALIZATION_FP_)


def load_model_and_normalization():
    """
    Loads the trained model and normalization values from disk.
    Returns:
        model: The loaded CostModel instance (in eval mode)
        input_max: Tensor for input normalization
        target_max: Tensor for target normalization
    """
    model = pt_cost_model.CostModel(num_layers=3, input_size=4)
    model.load_state_dict(torch.load(WEIGHT_FP_))
    model.eval()
    norms = torch.load(NORMALIZATION_FP_)
    input_max = norms["input_max"]
    target_max = norms["target_max"]
    logging.info(
        "Loaded input and target normalization values: %s, %s.", input_max, target_max
    )
    return model, input_max, target_max


def infer_model(
    model,
    input_max,
    target_max,
    total_ops: int,
    total_bytes: int,
    peak_gflops: float,
    peak_memory_bw_gbps: float,
) -> float:
    """
    Function to infer the cost using the trained model.
    :param model: Loaded CostModel instance
    :param input_max: Tensor for input normalization
    :param target_max: Tensor for target normalization
    :param total_ops: Total operations for the inference.
    :param total_bytes: Total bytes for the inference.
    :return: Predicted cost in nanoseconds.
    """
    input_data = torch.tensor(
        [[total_ops, total_bytes, peak_gflops, peak_memory_bw_gbps]],
        dtype=torch.float32,
    )
    input_data_norm = input_data / input_max
    with torch.no_grad():
        predicted_cost = model(input_data_norm) * target_max
    return predicted_cost.item()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run script in training or inference mode."
    )
    parser.add_argument(
        "--mode",
        choices=["training", "inference"],
        required=True,
        help="Mode to run: training or inference",
    )
    args = parser.parse_args()

    if args.mode == "training":
        logging.info("Running in training mode...")
        train_model()
    elif args.mode == "inference":
        logging.info("Running in inference mode...")
        peak_gflops = 100.0
        peak_memory_bw_gbps = 50.0
        roofline_model = RooflineModel(
            peak_gflops=peak_gflops, peak_memory_bw_gbps=peak_memory_bw_gbps
        )
        model, input_max, target_max = load_model_and_normalization()
        # Check results
        total_ops = 100
        total_bytes = 50
        predicted_cost = infer_model(
            model,
            input_max,
            target_max,
            total_ops,
            total_bytes,
            peak_gflops,
            peak_memory_bw_gbps,
        )
        actual_cost = roofline_model.get_cost_ns(total_ops, total_bytes)
        logging.info(
            "Predicted cost: %.2f ns, Actual cost: %.2f ns",
            predicted_cost,
            actual_cost,
        )


if __name__ == "__main__":
    main()
